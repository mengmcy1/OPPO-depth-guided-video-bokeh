"""Center-O1 and depth-safe 3-sigma footprints for camera-facing surfels."""
from dataclasses import replace

import torch

from src.aperture_renderer import GaussianScene
from src.ordered_layers import order_hidden


def footprint_scale(camera, tangent_sigma, normal_sigma, depth, K, minimum_sigma=.30,
                    eps2d=.0015, chunk=32768):
    """Find a safe tangent scale or drop at the minimum allowed footprint."""
    length=len(camera)
    sigma_factor=torch.ones(length,device=camera.device,dtype=camera.dtype)
    survive=torch.ones(length,device=camera.device,dtype=torch.bool)
    h,w=depth.shape
    for start in range(0,length,chunk):
        sl=slice(start,start+chunk)
        p=camera[sl]
        z=p[:,2].clamp_min(1e-6)
        center=(p@K.T)[:,:2]/z[:,None]
        xy=center.floor().long()
        v=K.diag()[:2]*tangent_sigma[sl]/z[:,None]
        q=-(p[:,:2]/z[:,None])*K.diag()[:2]*normal_sigma[sl,None]/z[:,None]
        radius=int(torch.ceil(3*(v.square()+q.square()+eps2d).sqrt().max()).item())+1
        axis=torch.arange(-radius,radius+1,device=p.device)
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        steps=torch.stack([xx.ravel(),yy.ravel()],-1)
        footprint_xy=xy[:,None]+steps[None]
        px,py=footprint_xy.unbind(-1)
        inside=(px>=0)&(px<w)&(py>=0)&(py<h)
        expected=depth[py.clamp(0,h-1), px.clamp(0,w-1)]
        intersects_bad=inside&torch.isfinite(expected)&(expected>0)&(z[:,None]+1e-5<1.03*expected)
        need=intersects_bad.any(-1)
        if not need.any():
            continue
        d=footprint_xy[need]+.5-center[need,None]
        dx,dy=d.unbind(-1)
        blocked=intersects_bad[need]
        tv=v[need].square()
        normal=q[need]
        covariance_xy=(normal[:,0]*normal[:,1])[:,None]
        sq=normal.square()+eps2d
        def allowed(scale):
            s=tv*scale[:,None].square()+sq
            a,b=s[:,0,None],s[:,1,None]
            det=(a*b-covariance_xy.square()).clamp_min(1e-12)
            chi2=(b*dx.square()+a*dy.square()-2*covariance_xy*dx*dy)/det
            return ~(blocked&(chi2<=9.)).any(-1)
        lo=(minimum_sigma/v[need].amin(-1).clamp_min(1e-8)).clamp(max=1.)
        hi=torch.ones_like(lo)
        must_shrink=~allowed(hi)
        ok_min=allowed(lo)
        for _ in range(9):
            mid=(lo+hi)/2
            safe=allowed(mid)
            lo=torch.where(safe,mid,lo)
            hi=torch.where(safe,hi,mid)
        kept=~must_shrink|ok_min
        scale=torch.where(must_shrink,lo,1.)
        sigma_factor[sl][need]=scale
        survive[sl][need]=kept
    return sigma_factor,survive


def apply_footprint_order(scene,visible_depth,K,minimum_sigma=.30):
    if scene is None:
        return None,dict(total=0,moved=0,shrunk=0,dropped=0)
    ordered,audit=order_hidden(scene,visible_depth,K,'O1')
    # Stage-1 Query uses camera-facing local covariances.
    if (ordered.quats[:,1:].abs()>1e-6).any():
        raise ValueError('Footprint order expects first-version camera-facing surfels')
    factor,keep=footprint_scale(ordered.means,ordered.scales[:,:2],ordered.scales[:,2],
                               visible_depth,K,minimum_sigma)
    fields={k:getattr(ordered,k)[keep] for k in ordered.__dataclass_fields__}
    fields['scales'][:,:2]*=factor[keep,None]
    return GaussianScene(**fields),dict(audit,total=len(factor),shrunk=int((keep&(factor<1)).sum()),
                                       dropped=int((~keep).sum()))


@torch.no_grad()
def push_behind_split_background(scene, split, background_depth, K, *, chunk=4096, audit=None):
    """Extend O1 to first-pass background hits without dropping Gaussians.

    The split map adds a depth floor only on b=1. Check the original pinhole
    footprint as well as its center. Scaling by the depth ratio along with
    the move preserves the pinhole-projected center and footprint.
    """
    if scene is None or not len(scene.means) or not split.any():
        if audit is not None:
            audit.update(total=len(scene.means) if scene is not None else 0,moved=0,
                         depth_ratio=[],center_depth_ratio=[],scale_ratio=[])
        return scene
    from gsplat.cuda._wrapper import fully_fused_projection

    reference=torch.where(split,background_depth,0.)
    height,width=reference.shape
    camera=torch.eye(4,device=K.device,dtype=K.dtype)
    projected=fully_fused_projection(scene.means,None,scene.quats,scene.scales,camera[None],K[None],
        width,height,eps2d=.0015,near_plane=1e-4,far_plane=1e5,calc_compensations=True)
    radii,means,depth,conics,compensations=(value[0] for value in projected)
    ordered,_=order_hidden(scene,reference,K,'O1')
    target=ordered.means[:,2].clone()
    binding=torch.where(target>scene.means[:,2],target/1.03,0.) if audit is not None else None
    for start in range(0,len(depth),chunk):
        sl=slice(start,start+chunk)
        radius=int(radii[sl].max().item())
        if radius<=0:
            continue
        axis=torch.arange(-radius-1,radius+2,device=K.device)
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        offset=torch.stack((xx.flatten(),yy.flatten()),-1)
        xy=means[sl].floor().long()[:,None]+offset[None]
        x,y=xy.unbind(-1)
        inside=(x>=0)&(x<width)&(y>=0)&(y<height)
        delta=means[sl,None]-xy-.5
        dx,dy=delta.unbind(-1)
        a,b,c=conics[sl].unbind(-1)
        power=.5*(a[:,None]*dx.square()+c[:,None]*dy.square())+b[:,None]*dx*dy
        limit=(scene.opacities[sl]*compensations[sl])[:,None]*torch.exp(-power)
        hit=inside&(radii[sl]>0).all(-1)[:,None]&(power>=0)&(limit>=1/255)
        z=reference[y.clamp(0,height-1),x.clamp(0,width-1)]
        needed=torch.where(hit&torch.isfinite(z),z,0.).amax(-1)
        target[sl]=torch.maximum(target[sl],needed*1.03)
        if binding is not None:
            binding[sl]=torch.maximum(binding[sl],needed)
    moved=target>scene.means[:,2]
    scale_ratio=torch.where(moved,target/scene.means[:,2].clamp_min(1e-8),1.)
    means=scene.means.clone()
    means[moved,:2]*=scale_ratio[moved,None]
    means[moved,2]=target[moved]
    if audit is not None:
        projected=scene.means[moved]@K.T
        xy=(projected[:,:2]/projected[:,2:]).floor().long()
        px,py=xy.unbind(-1)
        inside=(px>=0)&(px<width)&(py>=0)&(py<height)
        center_depth=reference[py.clamp(0,height-1),px.clamp(0,width-1)]
        own=inside&(center_depth>0)&torch.isfinite(center_depth)
        audit.update(total=len(depth),moved=int(moved.sum()),
                     depth_ratio=(target[moved]/binding[moved]).cpu().tolist(),
                     center_depth_ratio=(target[moved][own]/center_depth[own]).cpu().tolist(),
                     scale_ratio=scale_ratio[moved].cpu().tolist())
    return replace(scene,means=means,scales=scene.scales*scale_ratio[:,None])
