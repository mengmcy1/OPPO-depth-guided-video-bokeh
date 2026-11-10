"""C1 physical CoC mapping and pooled/patch metrics, independent of MoVieS.

The frozen mapping uses signed *radius*, not diameter, in pixel units.
Foreground (z < focus) is positive, as in the official BokehMe disparity demo.
"""
import math
from pathlib import Path
import numpy as np


def physical_coc_radius(depth, fx, aperture_radius_m, focus_m):
    depth=np.asarray(depth,dtype=np.float32)
    valid=np.isfinite(depth)&(depth>0)
    signed=np.zeros_like(depth)
    signed[valid]=float(fx)*float(aperture_radius_m)*(1/depth[valid]-1/float(focus_m))
    return signed,valid


def pipeline_defocus(depth, fx, aperture_radius_m, focus_m, defocus_scale=10.):
    radius,valid=physical_coc_radius(depth,fx,aperture_radius_m,focus_m)
    return radius/float(defocus_scale),valid


def plane_impulse_checks():
    fx=400.;aperture=.03;focus=2.;depth=np.array([1.,2.,4.],np.float32)
    radius,valid=physical_coc_radius(depth,fx,aperture,focus)
    control,_=pipeline_defocus(depth,fx,aperture,focus)
    assert np.allclose(radius,[6.,0.,-3.]) and valid.all()
    assert np.allclose(control*10,radius)
    # A point on each plane is projected through a disk of subcameras whose
    # principal-point shift aligns the axial focus plane. Pixel displacement is
    # minus signed radius times the unit aperture coordinate.
    count=4096;i=np.arange(count//2);r=np.sqrt((i+.5)/(count//2));t=i*math.pi*(3-math.sqrt(5))
    xy=np.stack([r*np.cos(t),r*np.sin(t)],-1);xy=np.concatenate([xy,-xy])
    displacement=fx*aperture*xy[:,None,:]*(1/focus-1/depth)[None,:,None]
    assert np.max(np.abs(displacement+xy[:,None,:]*radius[None,:,None]))<1e-5
    second=np.mean(np.sum(displacement**2,-1),0)
    assert np.allclose(second,radius**2/2,atol=1e-5)
    # The source's tanh soft disk reaches half its central weight at R. Its
    # support has a soft tail; the renderer is not identical to an ideal disk.
    x=np.linspace(0,10,10001);R=4.;profile=.5+.5*np.tanh(4*(R-x))
    half_radius=float(x[np.argmin(abs(profile-profile[0]/2))])
    assert abs(half_radius-R)<.002
    return dict(depth_m=depth.tolist(),focus_m=focus,fx=fx,aperture_radius_m=aperture,
                signed_radius_px=radius.tolist(),pipeline_control=control.tolist(),
                radius_second_moment_px2=second.tolist(),soft_disk_half_max_radius_px=half_radius,
                diameter_px=2*R,zero_focus_radius=0.,checks_passed=True,
                scope='Analytic point/plane projection and source soft-disk formula; runtime CUDA impulse check separate.')


def region_sse(prediction,target,mask):
    mask=np.asarray(mask,bool)
    valid=mask&np.isfinite(prediction).all(-1)&np.isfinite(target).all(-1)
    n=int(valid.sum());sse=float(np.sum((prediction[valid].astype(np.float64)-target[valid])**2))
    return dict(pixels=n,sse=sse,psnr=-10*math.log10(max(sse/(3*n),1e-12)) if n else None,
                excluded_nonfinite_pixels=int(mask.sum())-n)


def patch_metrics(prediction,target,mask,lpips_model,patch_size=64,device='cpu'):
    """Score complete fixed-grid patches, weighted by regional support.

    Mask-exterior pixels are never zeroed. LPIPS/SSIM see the whole patch, so
    this is a contextual patch metric, not a pixel-local masked LPIPS.
    """
    import torch
    import cv2
    def ssim(a,b):
        def local(x):return cv2.GaussianBlur(x,(11,11),1.5,borderType=cv2.BORDER_REFLECT)
        ua,ub=local(a),local(b)
        va,vb,cov=local(a*a)-ua*ua,local(b*b)-ub*ub,local(a*b)-ua*ub
        score=((2*ua*ub+.01**2)*(2*cov+.03**2)/
               ((ua*ua+ub*ub+.01**2)*(va+vb+.03**2)))
        return float(score[5:-5,5:-5].mean())
    h,w=mask.shape;positions=[];supports=[];images=[];truths=[];ssims=[]
    for y in range(0,h,patch_size):
        for x in range(0,w,patch_size):
            region=mask[y:y+patch_size,x:x+patch_size];n=int(region.sum())
            if not n:continue
            a=prediction[y:y+patch_size,x:x+patch_size];b=target[y:y+patch_size,x:x+patch_size]
            if a.shape!=(patch_size,patch_size,3):raise ValueError('Use an image resolution divisible by the fixed patch size')
            positions.append([y,x]);supports.append(n/(patch_size**2));images.append(a);truths.append(b)
            ssims.append(ssim(a,b))
    if not positions:return dict(patch_lpips=None,patch_ssim=None,patch_count=0,patch_support_ratios=[],status='empty_region')
    weight=np.asarray(supports,dtype=np.float64);weight/=weight.sum();values=[]
    with torch.inference_mode():
        for start in range(0,len(images),16):
            a=torch.from_numpy(np.stack(images[start:start+16])).permute(0,3,1,2).to(device)
            b=torch.from_numpy(np.stack(truths[start:start+16])).permute(0,3,1,2).to(device)
            values.extend(lpips_model(2*a-1,2*b-1).reshape(-1).cpu().numpy().tolist())
    return dict(patch_lpips=float(np.dot(weight,values)),patch_ssim=float(np.dot(weight,ssims)),
                patch_count=len(positions),patch_positions_yx=positions,patch_support_ratios=supports,
                patch_lpips_scores=values,status='full_patch_scores_weighted_by_region_support')


def region_protocol(regions,frame_id,depth,K,parameters):
    indices=np.flatnonzero(regions['frame_ids']==frame_id)
    if len(indices)!=1:raise ValueError('Matched oracle frame must occur exactly once in region file')
    t=int(indices[0]);radius,valid=physical_coc_radius(depth,K[0,0],parameters['aperture_radius_m'],parameters['focus_distance_m'])
    masks=dict(full_image=np.ones(depth.shape,bool),boundary=regions['D_t'][t],
               in_focus=valid&(abs(radius)<=1.),out_of_focus=valid&(abs(radius)>=2.),
               M_C1_proxy=regions['M_C1_proxy'][t],dynamic_unrecoverable=regions['dynamic_unrecoverable'][t],
               static_static=regions['static_static'][t])
    return masks,radius,valid
