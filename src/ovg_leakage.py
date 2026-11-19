"""Depth-sorted diagnostics for Query contributions on sampled optical rays."""
import math

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _product(a,b):
    return a*b


@tr.jit
def _contributions(M,C,Z,O,S,OFFSET,IDS,LIMIT,WEIGHT,FRONT,W:tl.constexpr,H:tl.constexpr,
                   GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0)
    p=tl.arange(0,256)
    px=(tile%GW)*16+p%16
    py=(tile//GW)*16+p//16
    good=(px<W)&(py<H)
    pix=py*W+px
    limit=tl.load(LIMIT+pix,good,other=0.)
    base=tl.load(OFFSET+tile)
    end=tl.load(OFFSET+tile+1)
    T=tl.full((256,),1.,tl.float32)
    total=tl.full((256,),0.,tl.float32)
    front=tl.full((256,),0.,tl.float32)
    while base<end:
        a=base+tl.arange(0,B)
        active=a<end
        idx=tl.load(IDS+a,active,other=0)
        mx=tl.load(M+2*idx,active,other=0)
        my=tl.load(M+2*idx+1,active,other=0)
        c0=tl.load(C+3*idx,active,other=0)
        c1=tl.load(C+3*idx+1,active,other=0)
        c2=tl.load(C+3*idx+2,active,other=0)
        z=tl.load(Z+idx,active,other=0)
        opacity=tl.load(O+idx,active,other=0)
        source=tl.load(S+idx,active,other=-1)
        dx=mx[None,:]-px[:,None]-.5
        dy=my[None,:]-py[:,None]-.5
        power=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        alpha=tl.minimum(.999,opacity[None,:]*tl.exp(-power))
        alpha=tl.where(active[None,:]&good[:,None]&(power>=0)&(alpha>=1/255),alpha,0.)
        remaining=1-alpha
        after=tl.associative_scan(remaining,1,_product)*T[:,None]
        query=tl.where((source[None,:]==1)&(after>1e-4),alpha*after/remaining,0.)
        total+=tl.sum(query,1)
        front+=tl.sum(tl.where(z[None,:]<limit[:,None],query,0.),1)
        T=tl.min(after,1)
        base+=B
    tl.store(WEIGHT+pix,total,good)
    tl.store(FRONT+pix,front,good)


@torch.no_grad()
def query_contribution(scene,K,view,limit):
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode
    h,w=limit.shape
    gw,gh=math.ceil(w/16),math.ceil(h/16)
    radii,m,z,conic,aa=fully_fused_projection(scene.means,None,scene.quats,scene.scales,view[None],K[None],
        w,h,eps2d=.0015,near_plane=1e-4,far_plane=1e5,calc_compensations=True)
    _,ids,flat=isect_tiles(m,radii,z,16,gw,gh)
    offsets=isect_offset_encode(ids,1,gw,gh).flatten()
    offsets=torch.cat((offsets,offsets.new_tensor([len(flat)])))
    wq,prefix=torch.empty_like(limit),torch.empty_like(limit)
    opacity=(scene.opacities[None]*aa).contiguous()
    _contributions[(gw*gh,)](m.contiguous(),conic.contiguous(),z.contiguous(),opacity,
        scene.source_frames.contiguous(),offsets,flat,limit.contiguous(),wq,prefix,w,h,gw,num_warps=8)
    return wq,prefix


@tr.jit
def _predicted_surfaces(M,C,Z,O,OFFSET,IDS,MASK,QW,QZ,VZ,VW,NV,NQ0,
                        W:tl.constexpr,H:tl.constexpr,GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0)
    p=tl.arange(0,256)
    px=(tile%GW)*16+p%16
    py=(tile//GW)*16+p//16
    good=(px<W)&(py<H)
    pix=py*W+px
    use=tl.load(MASK+pix,good,other=False)
    base=tl.load(OFFSET+tile)
    end=tl.load(OFFSET+tile+1)
    T=tl.full((256,),1.,tl.float32)
    Tv=tl.full((256,),1.,tl.float32)
    sum_q=tl.full((256,),0.,tl.float32)
    sum_zq=tl.full((256,),0.,tl.float32)
    max_v=tl.full((256,),0.,tl.float32)
    depth_v=tl.full((256,),0.,tl.float32)
    if tl.sum(use.to(tl.int32),0)>0:
        while base<end:
            a=base+tl.arange(0,B)
            active=a<end
            idx=tl.load(IDS+a,active,other=0)
            mx=tl.load(M+2*idx,active,other=0)
            my=tl.load(M+2*idx+1,active,other=0)
            c0=tl.load(C+3*idx,active,other=0)
            c1=tl.load(C+3*idx+1,active,other=0)
            c2=tl.load(C+3*idx+2,active,other=0)
            z=tl.load(Z+idx,active,other=0)
            opacity=tl.load(O+idx,active,other=0)
            dx=mx[None,:]-px[:,None]-.5
            dy=my[None,:]-py[:,None]-.5
            power=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
            alpha=tl.minimum(.999,opacity[None,:]*tl.exp(-power))
            alpha=tl.where(active[None,:]&use[:,None]&(power>=0)&(alpha>=1/255),alpha,0.)
            remaining=1-alpha
            after=tl.associative_scan(remaining,1,_product)*T[:,None]
            query=tl.where((idx[None,:]>=NQ0)&(after>1e-4),alpha*after/remaining,0.)
            sum_q+=tl.sum(query,1)
            sum_zq+=tl.sum(query*z[None,:],1)
            T=tl.min(after,1)
            av=tl.where(idx[None,:]<NV,alpha,0.)
            rv=1-av
            after_v=tl.associative_scan(rv,1,_product)*Tv[:,None]
            vw=tl.where(after_v>1e-4,av*after_v/rv,0.)
            local_max=tl.max(vw,1)
            pos=tl.argmax(vw,1,tie_break_left=True)
            at_max=tl.sum(tl.where(tl.arange(0,B)[None,:]==pos[:,None],z[None,:],0.),1)
            depth_v=tl.where(local_max>max_v,at_max,depth_v)
            max_v=tl.maximum(max_v,local_max)
            Tv=tl.min(after_v,1)
            base+=B
    tl.store(QW+pix,sum_q,good)
    tl.store(QZ+pix,tl.where(sum_q>0,sum_zq/sum_q,0.),good)
    tl.store(VZ+pix,depth_v,good)
    tl.store(VW+pix,max_v,good)


@torch.no_grad()
def predicted_surfaces(merged,K,view,mask,visible_count,query_start):
    """Return Query weight/depth in the full composite and dominant visible-only splat.

    The concatenated layout is visible, first-pass background, second-pass
    Query. The visible weights use a separate transmittance: a Query in front
    must not erase the visible surface used to detect that ordering error.
    A zero depth means the corresponding layer made no contribution.
    """
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode
    h,w=mask.shape
    gw,gh=math.ceil(w/16),math.ceil(h/16)
    radii,m,z,conic,aa=fully_fused_projection(merged.means,None,merged.quats,merged.scales,
        view[None],K[None],w,h,eps2d=.0015,near_plane=1e-4,far_plane=1e5,calc_compensations=True)
    _,ids,flat=isect_tiles(m,radii,z,16,gw,gh)
    offsets=isect_offset_encode(ids,1,gw,gh).flatten()
    offsets=torch.cat((offsets,offsets.new_tensor([len(flat)])))
    out=tuple(torch.empty(mask.shape,device=mask.device,dtype=merged.means.dtype) for _ in range(4))
    _predicted_surfaces[(gw*gh,)](m.contiguous(),conic.contiguous(),z.contiguous(),
        (merged.opacities[None]*aa).contiguous(),offsets,flat,mask.contiguous(),*out,
        visible_count,query_start,w,h,gw,num_warps=8)
    return out
