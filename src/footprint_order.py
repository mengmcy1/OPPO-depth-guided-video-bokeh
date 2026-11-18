"""Center-O1 and depth-safe 3-sigma footprints for camera-facing surfels."""
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
