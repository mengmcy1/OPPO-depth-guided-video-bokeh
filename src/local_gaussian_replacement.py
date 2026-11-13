"""Fixed, local output-Gaussian suppression diagnostics (no network updates).

Observed weights use gsplat's exact front-to-back hit enumerator. Support
intersection uses unoccluded projected footprints with the same AA compensation,
alpha cap and cutoff; using only Gaussian center pixels would miss footprints.
"""
import math
import torch
import triton as tr
import triton.language as tl
from .aperture_renderer import GaussianScene


@tr.jit
def _weights(GI,START,COUNT,PIX,M,C,O,MAXIMUM,SUM,W:tl.constexpr,L:tl.constexpr):
    pixel=tl.load(PIX+tl.program_id(0));start=tl.load(START+pixel);n=tl.load(COUNT+pixel)
    j=tl.arange(0,L);idx=tl.load(GI+start+j,j<n,other=0)
    dx=tl.load(M+2*idx)-pixel%W-.5;dy=tl.load(M+2*idx+1)-pixel//W-.5
    sigma=.5*(tl.load(C+3*idx)*dx*dx+tl.load(C+3*idx+2)*dy*dy)+tl.load(C+3*idx+1)*dx*dy
    a=tl.where(j<n,tl.minimum(.999,tl.load(O+idx)*tl.exp(-sigma)),0.)
    weight=a*tl.cumprod(1-a,0)/tl.maximum(1-a,1e-8)
    tl.atomic_max(MAXIMUM+idx,weight,j<n,sem='relaxed')
    tl.atomic_add(SUM+idx,weight,j<n,sem='relaxed')


@torch.no_grad()
def projection(scene,K,view=None,width=512,height=512):
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode
    if view is None:view=torch.eye(4,device=K.device,dtype=K.dtype)
    gw,gh=math.ceil(width/16),math.ceil(height/16)
    r,m,z,c,comp=fully_fused_projection(scene.means,None,scene.quats,scene.scales,view[None],K[None],width,height,
        eps2d=.0015,near_plane=1e-4,far_plane=1e5,calc_compensations=True)
    o=(scene.opacities[None]*comp).contiguous();_,ii,flat=isect_tiles(m,r,z,16,gw,gh)
    offset=isect_offset_encode(ii,1,gw,gh)
    return m.contiguous(),z.contiguous(),c.contiguous(),o,offset,flat


@torch.no_grad()
def input_observation_weights(scene,K,width=512,height=512):
    """max_p w_i(p), sum_p w_i(p), in this Gaussian cloud's target input view."""
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range
    m,z,c,o,offset,flat=projection(scene,K,width=width,height=height)
    gi,pi,_=rasterize_to_indices_in_range(0,100000,torch.ones((1,height,width),device=K.device),m,c,o,width,height,16,offset,flat)
    if not bool((pi[1:]>=pi[:-1]).all()):raise RuntimeError('Hit enumerator no longer groups pixels')
    count=torch.bincount(pi.long(),minlength=height*width).contiguous();start=(count.cumsum(0)-count).contiguous()
    maximum=torch.zeros(len(scene.means),device=K.device);mass=torch.zeros_like(maximum)
    last=0;largest=int(count.max())
    for exponent in range(4,max(4,math.ceil(math.log2(max(largest,1))))+1):
        length=2**exponent;pixels=torch.nonzero((count>last)&(count<=length)).flatten().contiguous()
        if len(pixels):_weights[(len(pixels),)](gi,start,count,pixels,m,c,o,maximum,mass,width,length,num_warps=4 if length<=128 else 8)
        last=length
    return maximum,mass,dict(accepted_hits=len(gi),max_hits_per_pixel=largest,weights='native front-to-back alpha contributions,including native cutoff and earlystop',input_view='target frame center camera only;not the other12 cameras')


@tr.jit
def _support(M,C,Z,O,S,OFFSET,IDS,SUPPORT,VISIBLE,HIDDEN,UNOBS,FIRST,SECOND,W:tl.constexpr,H:tl.constexpr,GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0);p=tl.arange(0,256);px=tile%GW*16+p%16;py=tile//GW*16+p//16
    good=(px<W)&(py<H);support=tl.load(SUPPORT+py*W+px,good,other=0)&good
    visible=tl.load(VISIBLE+py*W+px,support,other=0.);hidden=tl.load(HIDDEN+py*W+px,support,other=0.)
    start=tl.load(OFFSET+tile);end=tl.load(OFFSET+tile+1);base=start
    if tl.sum(support.to(tl.int32),0)>0:
      while base<end:
        j=base+tl.arange(0,B);inside=j<end;idx=tl.load(IDS+j,inside,other=0)
        dx=tl.load(M+idx*2,inside,other=0)[None,:]-px[:,None]-.5;dy=tl.load(M+idx*2+1,inside,other=0)[None,:]-py[:,None]-.5
        c0=tl.load(C+idx*3,inside,other=0);c1=tl.load(C+idx*3+1,inside,other=0);c2=tl.load(C+idx*3+2,inside,other=0)
        sigma=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        alpha=tl.minimum(.999,tl.load(O+idx,inside,other=0)[None,:]*tl.exp(-sigma))
        hit=inside[None,:]&support[:,None]&(sigma>=0)&(alpha>=1/255)
        depth=tl.load(Z+idx,inside,other=0);layer=tl.load(S+idx,inside,other=0)
        r1=hit&(layer[None,:]==1)&(depth[None,:]>visible[:,None])&(hidden[:,None]>0)&(tl.abs(depth[None,:]-hidden[:,None])<=.2*hidden[:,None])
        r2=hit&tl.load(UNOBS+idx,inside,other=0)[None,:]
        tl.atomic_max(FIRST+idx,1,inside&(tl.sum(r1.to(tl.int32),0)>0),sem='relaxed')
        tl.atomic_max(SECOND+idx,1,inside&(tl.sum(r2.to(tl.int32),0)>0),sem='relaxed')
        base+=B


@torch.no_grad()
def local_suppression_masks(scene,K,support,visible_depth,hidden_depth,unobserved):
    h,w=support.shape;m,z,c,o,offset,flat=projection(scene,K,width=w,height=h)
    off=torch.cat([offset.flatten(),torch.tensor([len(flat)],device=K.device,dtype=offset.dtype)])
    r1=torch.zeros(len(scene.means),device=K.device,dtype=torch.int32);r2=torch.zeros_like(r1)
    _support[(math.ceil(w/16)*math.ceil(h/16),)](m,c,z,o,scene.source_frames,off,flat,support.contiguous(),visible_depth.contiguous(),hidden_depth.contiguous(),unobserved.contiguous(),r1,r2,w,h,math.ceil(w/16),num_warps=8)
    return r1.bool(),r2.bool()


def suppress(scene,mask):
    """Preserve positions, sizes, rotations, colors and stable Gaussian indices."""
    return GaussianScene(scene.means,scene.quats,scene.scales,torch.where(mask,0.,scene.opacities),scene.colors,scene.source_frames)


@torch.no_grad()
def blocker_categories_projected(scene,projected,depth_limit,selection,drop=None):
    """Reuse one projection for all matrix rows; same six bins as appendix9.

    Full products, no native early stop. The mask changes opacity only, so
    projection and tile intersections can be shared without changing results.
    """
    from .blocker_attribution import _categories
    m,z,c,o,offset,flat=projected
    if drop is not None:o=(o*(~drop)[None]).contiguous()
    h,w=depth_limit.shape;gw,gh=math.ceil(w/16),math.ceil(h/16)
    off=torch.cat([offset.flatten(),torch.tensor([len(flat)],device=offset.device,dtype=offset.dtype)])
    out=torch.empty((6,h,w),device=depth_limit.device)
    _categories[(gw*gh,)](m,c,z,o,scene.source_frames.contiguous(),off,flat,depth_limit.contiguous(),selection.contiguous(),out,w,h,gw,num_warps=8)
    return out


@tr.jit
def _ray_prefix(M,C,Z,O,S,OFFSET,IDS,UV,LIMIT,QORDER,TILE,QBEGIN,QEND,OUT,Q,B:tl.constexpr=32):
    job=tl.program_id(0);tile=tl.load(TILE+job);begin=tl.load(QBEGIN+job);endq=tl.load(QEND+job)
    j=begin+tl.arange(0,64);good=j<endq;ray=tl.load(QORDER+j,good,other=0)
    px=tl.load(UV+2*ray,good,other=0.);py=tl.load(UV+2*ray+1,good,other=0.);limit=tl.load(LIMIT+ray,good,other=0.)
    start=tl.load(OFFSET+tile);end=tl.load(OFFSET+tile+1);base=start;logs=tl.full((8,64),0.,tl.float32)
    while base<end:
        k=base+tl.arange(0,B);inside=k<end;idx=tl.load(IDS+k,inside,other=0)
        dx=tl.load(M+2*idx,inside,other=0)[None,:]-px[:,None];dy=tl.load(M+2*idx+1,inside,other=0)[None,:]-py[:,None]
        c0=tl.load(C+3*idx,inside,other=0);c1=tl.load(C+3*idx+1,inside,other=0);c2=tl.load(C+3*idx+2,inside,other=0)
        sigma=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        alpha=tl.minimum(.999,tl.load(O+idx,inside,other=0)[None,:]*tl.exp(-sigma))
        z=tl.load(Z+idx,inside,other=0);source=tl.load(S+idx,inside,other=0)
        use=good[:,None]&inside[None,:]&(sigma>=0)&(alpha>=1/255)&(z[None,:]<limit[:,None])
        margin=(limit[:,None]-z[None,:])/tl.maximum(limit[:,None],1e-8)
        category=source[None,:]*3+tl.where(margin<=.05,0,tl.where(margin<=.2,1,2))
        for cat in tl.static_range(6):
            value=tl.sum(tl.where(use&(category==cat),-tl.log(1-alpha),0.),1)
            logs+=tl.where(tl.arange(0,8)[:,None]==cat,value[None,:],0.)
        base+=B
    for cat in tl.static_range(6):
        value=tl.sum(tl.where(tl.arange(0,8)[:,None]==cat,logs,0.),0)
        tl.store(OUT+cat*Q+ray,value,good)


def ray_query_plan(uv,width=512,height=512):
    """Exact continuous query locations, grouped by their actual raster tile."""
    uv=uv.contiguous();valid=torch.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    ix=torch.nonzero(valid).flatten();tile=(uv[ix,1].floor().long()//16)*math.ceil(width/16)+uv[ix,0].floor().long()//16
    order=torch.argsort(tile,stable=True);tiles,counts=torch.unique_consecutive(tile[order],return_counts=True);starts=counts.cumsum(0)-counts
    jobtile=[];begin=[];end=[]
    for tt,ss,nn in zip(tiles.cpu().tolist(),starts.cpu().tolist(),counts.cpu().tolist()):
        for a in range(ss,ss+nn,64):jobtile.append(tt);begin.append(a);end.append(min(a+64,ss+nn))
    cv=lambda a:torch.tensor(a,device=uv.device,dtype=torch.int64)
    return (uv,ix[order].contiguous(),cv(jobtile),cv(begin),cv(end),valid)


@torch.no_grad()
def blocker_ray_queries(scene,projected,depth_limit,plan,drop=None):
    """Full prefix at exact rays through placed parent points, no rounding.

    This avoids attaching nominal K16 pixel labels to a different ray when
    focus or hidden-depth placement changes. Invalid/out-of-frame rays are
    flagged in plan[5], never silently interpreted as transparent.
    """
    m,z,c,o,offset,flat=projected;uv,order,tile,begin,end,valid=plan
    if drop is not None:o=(o*(~drop)[None]).contiguous()
    off=torch.cat([offset.flatten(),torch.tensor([len(flat)],device=offset.device,dtype=offset.dtype)])
    out=torch.zeros((6,len(uv)),device=uv.device)
    if len(tile):_ray_prefix[(len(tile),)](m,c,z,o,scene.source_frames.contiguous(),off,flat,uv,depth_limit.contiguous(),order,tile,begin,end,out,len(uv),num_warps=4)
    return out
