"""Differentiable finite-aperture quadrature with a fixed residual moment.

The K64 default is ordinary thin-lens quadrature. The optional covariance
mode is a moment approximation for training, not an exact optical renderer.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .aperture_renderer import aperture_disk


def lens_points(samples,aperture,device,dtype,residual=False,seed=None):
    if seed is None:points=aperture_disk(samples,aperture,device=device,dtype=dtype)
    else:
        if samples!=16:raise ValueError('Registered random strata require K16')
        generator=torch.Generator(device=device).manual_seed(seed);u=torch.rand(4,2,2,device=device,dtype=dtype,generator=generator)
        radial=(torch.arange(4,device=device,dtype=dtype)[:,None]+u[...,0])/4
        angle=(torch.arange(2,device=device,dtype=dtype)[None,:]+u[...,1])*math.pi/2
        first=torch.stack([radial.sqrt()*angle.cos(),radial.sqrt()*angle.sin()],-1).reshape(-1,2)
        points=torch.cat([first,-first],0)*aperture
    if residual:points=points*math.sqrt(1-1/samples)
    return points


def render_residual(scene,K,E,width,height,focus,aperture,samples,fill,bgdepth,
                    residual=False,seed=None,checkpoint_rays=False,eps2d=.0015,far_plane=1e4):
    from gsplat import rasterization
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci,proj
    disk=lens_points(samples,aperture,K.device,K.dtype,residual,seed)
    means=scene.means;covars=None;cov_new=None;variance=None
    if residual:
        cov_world=quat_scale_to_covar_preci(scene.quats,scene.scales,compute_covar=True,compute_preci=False,triu=False)[0]
        rotation=E[:3,:3];covars=rotation@cov_world@rotation.T
        z=(means@rotation.T+E[:3,3])[:,2].clamp_min(1e-4);radius=K[0,0]*aperture*(1/z-1/focus);variance=radius.square()/(4*samples)
        extra=torch.zeros_like(covars);extra[:,0,0]=variance*(z/K[0,0]).square();extra[:,1,1]=variance*(z/K[1,1]).square()
        cov_new=cov_world+rotation.T@extra@rotation
    y,x=torch.meshgrid(torch.arange(height,device=K.device),torch.arange(width,device=K.device),indexing='ij')
    pixel=torch.stack([x+.5,y+.5],-1);zbg=torch.where(bgdepth>0,bgdepth,torch.full_like(bgdepth,1e4))
    def ray(means,quats,scales,opacities,colors,base_cov,expanded_cov,variance,view,intr,xy):
        uv=pixel+xy*torch.stack([K[0,0],K[1,1]])*(1/zbg-1/focus)[...,None]
        grid=torch.stack([2*uv[...,0]/width-1,2*uv[...,1]/height-1],-1)
        colorfill=F.grid_sample(fill.permute(2,0,1)[None],grid[None],padding_mode='border',align_corners=False)[0].permute(1,2,0)
        op=opacities
        if residual:
            camera=means@view[:3,:3].T+view[:3,3]
            _,projected_cov=proj(camera[None],base_cov[None],intr[None],width,height,camera_model='pinhole')
            xy_cov=projected_cov[0]
            # Explicit 2x2 determinants avoid launching batched LU solves for
            # millions of small matrices; this is the same area formula.
            xx=xy_cov[:,0,0];yy=xy_cov[:,1,1];cross=xy_cov[:,0,1]*xy_cov[:,1,0]
            det_old=(xx*yy-cross).clamp_min(1e-20)
            det_new=((xx+variance)*(yy+variance)-cross).clamp_min(1e-20)
            op=opacities*(det_old/det_new).clamp(0,1).sqrt()
        image,a,_=rasterization(means,None if residual else quats,None if residual else scales,op,colors,view[None],intr[None],width,height,
            packed=False,render_mode='RGB+ED',rasterize_mode='antialiased',near_plane=1e-4,far_plane=far_plane,eps2d=eps2d,covars=expanded_cov)
        alpha=a[0,...,0]
        return image[0,...,:3]+(1-alpha[...,None])*colorfill,alpha
    rgb=torch.zeros_like(fill);alpha=torch.zeros_like(bgdepth)
    for xy in disk:
        # Residual covariance and camera rotation are invariant to lens-plane
        # translation; hoist them once while preserving the same ray geometry.
        view=E.clone()
        view[:2,3]-=xy;intr=K.clone();intr[0,2]+=K[0,0]*xy[0]/focus;intr[1,2]+=K[1,1]*xy[1]/focus
        args=(means,scene.quats,scene.scales,scene.opacities,scene.colors,covars,cov_new,variance,view,intr,xy)
        if checkpoint_rays and torch.is_grad_enabled():color,a=checkpoint(ray,*args,use_reentrant=False)
        else:color,a=ray(*args)
        rgb=rgb+color;alpha=alpha+a
    return dict(rgb_linear=rgb/samples,alpha=alpha/samples,aperture_xy=disk)
