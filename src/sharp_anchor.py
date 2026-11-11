"""Target-RGB-only output-parameter anchoring; no SHARP network updates."""
import time
import numpy as np
import torch
from gsplat import rasterization
from .aperture_renderer import GaussianScene,linear_to_srgb
from .upper_bound_optimization import ssim


def zero_image(cloud,K,fill):
    image,alpha,_=rasterization(cloud.means,cloud.quats,cloud.scales,cloud.opacities,cloud.colors,
        torch.eye(4,device=K.device)[None],K[None],512,512,packed=False,render_mode='RGB+ED',
        rasterize_mode='antialiased',eps2d=.0015,near_plane=1e-4,far_plane=1e5)
    return linear_to_srgb(image[0,...,:3]+(1-alpha[0])*fill).clamp(0,1)


def anchor(cloud,K,fill,target,with_opacity,callback,steps=1000):
    """All supervision is supplied as the single current RGB tensor."""
    assert not any(x.requires_grad for x in [cloud.means,cloud.quats,cloud.scales,cloud.opacities,cloud.colors])
    colors=torch.nn.Parameter(cloud.colors.detach().clone());delta=torch.nn.Parameter(torch.zeros_like(cloud.opacities)) if with_opacity else None
    optimizer=torch.optim.Adam([colors]+([delta] if delta is not None else []),lr=.01,eps=1e-8)
    original_colors=cloud.colors;original_opacity=cloud.opacities
    def current():
        opacity=(original_opacity+.02*delta.tanh()).clamp(0,1) if delta is not None else original_opacity
        return GaussianScene(cloud.means,cloud.quats,cloud.scales,opacity,colors,cloud.source_frames)
    started=time.monotonic();rows=[];times=[];reason=None;completed_updates=0;bad_monitors=0
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps+1):
        if step in [0,50,100,250,500,750,1000]:
            with torch.inference_mode():
                pred=zero_image(current(),K,fill);mse=float((pred-target).square().double().mean())
                loss=float(.8*(pred-target).abs().mean()+.2*(1-ssim(pred,target)))
            if not np.isfinite(loss+mse):reason='nonfinite target-only monitor';break
            row=dict(step=step,input_PSNR_dB=float(-10*np.log10(max(mse,1e-12))),input_loss=loss,seconds=time.monotonic()-started,
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
            rows.append(row);callback(dict(completed=False,completed_updates=completed_updates,rows=rows))
            print('ANCHOR_MONITOR',with_opacity,step,row['input_PSNR_dB'],loss,flush=True)
            if step>0:bad_monitors=bad_monitors+1 if loss>5*rows[0]['input_loss'] else 0
            if bad_monitors>=3:reason='persistent target-only loss divergence';break
        if step==steps:break
        optimizer.param_groups[0]['lr']=.01*(.01+.99*.5*(1+np.cos(np.pi*step/steps)))
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
        pred=zero_image(current(),K,fill);loss=.8*(pred-target).abs().mean()+.2*(1-ssim(pred,target))
        if not bool(torch.isfinite(loss)):reason='nonfinite optimization loss';break
        loss.backward()
        if not bool(torch.isfinite(colors.grad).all()) or (delta is not None and not bool(torch.isfinite(delta.grad).all())):
            reason='nonfinite output-parameter gradient';break
        optimizer.step()
        with torch.no_grad():colors.clamp_(0,1)
        torch.cuda.synchronize();times.append(time.monotonic()-start);completed_updates+=1
    fitted=current()
    if reason is None and rows[-1]['input_PSNR_dB']<=rows[0]['input_PSNR_dB']:reason='final zero-aperture input PSNR did not improve'
    with torch.no_grad():
        change=(fitted.colors-original_colors).abs();opacity_change=(fitted.opacities-original_opacity).abs()
        stats=dict(color_changed_fraction=float((change.max(-1).values>1e-5).float().mean()),
            color_abs_delta_quantiles=torch.quantile(change.flatten(),torch.tensor([.1,.5,.9,.99,1.],device=change.device)).cpu().tolist(),
            Gaussian_RGB_max_delta_quantiles=torch.quantile(change.max(-1).values,torch.tensor([.1,.5,.9,.99,1.],device=change.device)).cpu().tolist(),
            opacity_changed_fraction=float((opacity_change>1e-5).float().mean()),opacity_max_abs_delta=float(opacity_change.max()))
        assert stats['opacity_max_abs_delta']<=.020001
    result=dict(completed=reason is None and completed_updates==steps,stop_reason=reason,completed_updates=completed_updates,rows=rows,
        wall_seconds=time.monotonic()-started,median_step_seconds=float(np.median(times)) if times else None,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,geometry_fixed=True,network_weights_updated=False,
        target_RGB_only=True,evaluation_data_read=False,**stats)
    return fitted,result
