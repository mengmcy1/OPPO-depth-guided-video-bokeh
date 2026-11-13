"""Diagnostic -log(T) by visible layer and distance in front of a hidden hit.

Dropping a category here is a transmittance statistic, not a rendered method.
No early termination: categories behind an opaque contributor are needed for
the counterfactual products. Projection/AA/alpha cutoffs match gsplat.
"""
import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _categories(M,C,Z,O,S,OFFSET,IDS,LIMIT,SELECT,OUT,W:tl.constexpr,H:tl.constexpr,
                GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0);p=tl.arange(0,256)
    px=(tile%GW)*16+p%16;py=(tile//GW)*16+p//16
    good=(px<W)&(py<H)
    good=good & tl.load(SELECT+py*W+px,good,other=0)
    limit=tl.load(LIMIT+py*W+px,good,other=0.)
    start=tl.load(OFFSET+tile);end=tl.load(OFFSET+tile+1)
    logs=tl.full((8,256),0.,tl.float32);base=start
    if tl.sum(good.to(tl.int32),0)>0:
      while base<end:
        a=base+tl.arange(0,B);inside=a<end;idx=tl.load(IDS+a,inside,other=0)
        mx=tl.load(M+2*idx,inside,other=0);my=tl.load(M+2*idx+1,inside,other=0)
        c0=tl.load(C+3*idx,inside,other=0);c1=tl.load(C+3*idx+1,inside,other=0);c2=tl.load(C+3*idx+2,inside,other=0)
        z=tl.load(Z+idx,inside,other=0);opacity=tl.load(O+idx,inside,other=0);source=tl.load(S+idx,inside,other=0)
        dx=mx[None,:]-px[:,None]-.5;dy=my[None,:]-py[:,None]-.5
        sigma=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        alpha=tl.minimum(.999,opacity[None,:]*tl.exp(-sigma))
        use=inside[None,:]&good[:,None]&(sigma>=0)&(alpha>=1/255)&(z[None,:]<limit[:,None])
        margin=(limit[:,None]-z[None,:])/tl.maximum(limit[:,None],1e-8)
        category=source[None,:]*3+tl.where(margin<=.05,0,tl.where(margin<=.2,1,2))
        for k in tl.static_range(6):
            contribution=tl.sum(tl.where(use&(category==k),-tl.log(1-alpha),0.),1)
            logs+=tl.where(tl.arange(0,8)[:,None]==k,contribution[None,:],0.)
        base+=B
    for k in tl.static_range(6):
        value=tl.sum(tl.where(tl.arange(0,8)[:,None]==k,logs,0.),0)
        tl.store(OUT+k*W*H+py*W+px,value,(px<W)&(py<H))


@torch.no_grad()
def blocker_log_categories(scene,K,view,depth_limit,selection):
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode
    if not bool(((scene.source_frames==0)|(scene.source_frames==1)).all()):
        raise ValueError('Attribution requires unchanged SHARP layer IDs 0/1')
    h,w=depth_limit.shape;gw,gh=math.ceil(w/16),math.ceil(h/16)
    radii,m,z,conics,comp=fully_fused_projection(scene.means,None,scene.quats,scene.scales,view[None],K[None],w,h,eps2d=.0015,near_plane=1e-4,far_plane=1e5,calc_compensations=True)
    opacity=(scene.opacities[None]*comp).contiguous()
    _,ids,flat=isect_tiles(m,radii,z,16,gw,gh);offset=isect_offset_encode(ids,1,gw,gh).reshape(-1)
    offset=torch.cat([offset,torch.tensor([len(flat)],device=offset.device,dtype=offset.dtype)])
    out=torch.empty((6,h,w),device=K.device,dtype=torch.float32)
    _categories[(gw*gh,)](m.contiguous(),conics.contiguous(),z.contiguous(),opacity,scene.source_frames.contiguous(),offset,flat,depth_limit.contiguous(),selection.contiguous(),out,w,h,gw,num_warps=8)
    return out
