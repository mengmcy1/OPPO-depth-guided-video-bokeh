"""Fixed-geometry color diagnostics: native alpha weights, no RGB renderer.

K16 identity statistics use continuous rays and source-pixel GT identities.
K64 predictions sum w_i*delta_color; native geometry/opacity never changes.
"""
import math
import torch
import triton as tr
import triton.language as tl
from .local_gaussian_replacement import projection


@tr.jit
def _surface(M,C,Z,O,SID,SLOCAL,CVALID,UNOBS,OFFSET,IDS,UV,QORDER,TILE,QBEGIN,QEND,
             TID,TLOCAL,TFOOT,RAW,ALIGNED,OUT,GMASS,Q,B:tl.constexpr=32):
    job=tl.program_id(0);tile=tl.load(TILE+job);begin=tl.load(QBEGIN+job);endq=tl.load(QEND+job)
    j=begin+tl.arange(0,64);good=j<endq;ray=tl.load(QORDER+j,good,other=0)
    px=tl.load(UV+2*ray,good,other=0.);py=tl.load(UV+2*ray+1,good,other=0.)
    target=tl.load(TID+ray,good,other=0);tx=tl.load(TLOCAL+3*ray,good,other=0.);ty=tl.load(TLOCAL+3*ray+1,good,other=0.);tz=tl.load(TLOCAL+3*ray+2,good,other=0.)
    tolerance=3*tl.load(TFOOT+ray,good,other=0.);raw=tl.load(RAW+ray,good,other=0.);aligned=tl.load(ALIGNED+ray,good,other=0.)
    T=tl.full((64,),1.,tl.float32);active=good
    alpha=tl.full((64,),0.,tl.float32);matched=alpha;available=alpha;main=alpha;close_raw=alpha;close_aligned=alpha;count=alpha;matched_count=alpha
    top=tl.full((64,32),0,tl.uint64);gather=tl.broadcast_to(tl.arange(0,32)[None,:],(64,32))
    base=tl.load(OFFSET+tile);end=tl.load(OFFSET+tile+1)
    while (base<end)&(tl.sum(active.to(tl.int32),0)>0):
        k=base+tl.arange(0,B);inside=k<end;idx=tl.load(IDS+k,inside,other=0)
        dx=tl.load(M+2*idx,inside,other=0)[None,:]-px[:,None];dy=tl.load(M+2*idx+1,inside,other=0)[None,:]-py[:,None]
        c0=tl.load(C+3*idx,inside,other=0);c1=tl.load(C+3*idx+1,inside,other=0);c2=tl.load(C+3*idx+2,inside,other=0)
        sigma=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        a=tl.minimum(.999,tl.load(O+idx,inside,other=0)[None,:]*tl.exp(-sigma))
        a=tl.where(inside[None,:]&active[:,None]&(sigma>=0)&(a>=1/255),a,0.)
        prefix=tl.cumprod(1-a,1);after=T[:,None]*prefix
        weight=tl.where(after>1e-4,T[:,None]*prefix/tl.maximum(1-a,1e-8)*a,0.)
        sx=tl.load(SLOCAL+3*idx,inside,other=0);sy=tl.load(SLOCAL+3*idx+1,inside,other=0);sz=tl.load(SLOCAL+3*idx+2,inside,other=0)
        lx=sx[None,:]-tx[:,None];ly=sy[None,:]-ty[:,None];lz=sz[None,:]-tz[:,None]
        same=(tl.load(SID+idx,inside,other=0)[None,:]==target[:,None])&(target[:,None]>0)&(lx*lx+ly*ly+lz*lz<=tolerance[:,None]*tolerance[:,None])
        has_color=tl.load(CVALID+idx,inside,other=0);latent=tl.load(UNOBS+idx,inside,other=0);zz=tl.load(Z+idx,inside,other=0)
        mass=weight*same;alpha+=tl.sum(weight,1);matched+=tl.sum(mass,1);available+=tl.sum(mass*has_color[None,:],1);main+=tl.sum(mass*(has_color&latent)[None,:],1)
        close_raw+=tl.sum(weight*((~same)&(tl.abs(zz[None,:]-raw[:,None])<=.05*raw[:,None])),1)
        close_aligned+=tl.sum(weight*((~same)&(aligned[:,None]>0)&(tl.abs(zz[None,:]-aligned[:,None])<=.05*aligned[:,None])),1)
        count+=tl.sum((weight>0).to(tl.float32),1);matched_count+=tl.sum(((weight>0)&same).to(tl.float32),1)
        tl.atomic_add(GMASS+idx,tl.sum(mass,0),inside,sem='relaxed')
        # Exact top8 (retain top32 between blocks); ties use Gaussian index,
        # independently of GT identity. Positive float bits preserve order.
        packed=(weight.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)|(0xffffffff-idx[None,:].to(tl.uint64))
        packed=tl.where(weight>0,packed,0)
        merged=tl.reshape(tl.join(top,packed),(64,64))
        top=tl.gather(tl.sort(merged,dim=1,descending=True),gather,1)
        T*=tl.sum(tl.where(tl.arange(0,B)[None,:]==B-1,prefix,0.),1);active&=T>1e-4;base+=B
    ix=(0xffffffff-(top&0xffffffff)).to(tl.int64);valid_top=(top>0)&(tl.arange(0,32)[None,:]<8)&good[:,None]
    sx=tl.load(SLOCAL+3*ix,valid_top,other=0.);sy=tl.load(SLOCAL+3*ix+1,valid_top,other=0.);sz=tl.load(SLOCAL+3*ix+2,valid_top,other=0.)
    lx=sx-tx[:,None];ly=sy-ty[:,None];lz=sz-tz[:,None]
    topmatch=valid_top&(tl.load(SID+ix,valid_top,other=0)==target[:,None])&(target[:,None]>0)&(lx*lx+ly*ly+lz*lz<=tolerance[:,None]*tolerance[:,None])
    topz=tl.load(Z+ix,valid_top,other=0.)
    for field in tl.static_range(12):
        value=alpha
        if field==1:value=matched
        elif field==2:value=available
        elif field==3:value=main
        elif field==4:value=close_raw
        elif field==5:value=close_aligned
        elif field==6:value=tl.sum(valid_top.to(tl.float32),1)
        elif field==7:value=tl.sum(topmatch.to(tl.float32),1)
        elif field==8:value=count
        elif field==9:value=matched_count
        elif field==10:value=tl.sum((valid_top&~topmatch&(aligned[:,None]>0)&(tl.abs(topz-aligned[:,None])<=.05*aligned[:,None])).to(tl.float32),1)
        elif field==11:value=tl.sum((valid_top&~topmatch&(tl.abs(topz-raw[:,None])<=.05*raw[:,None])).to(tl.float32),1)
        tl.store(OUT+field*Q+ray,value,good)


@torch.no_grad()
def surface_statistics(scene,projected,plan,target_id,target_local,foot,raw_depth,aligned_depth,source_id,source_local,color_valid,unobserved,gaussian_mass):
    m,z,c,o,offset,flat=projected;uv,order,tile,begin,end,valid=plan
    off=torch.cat([offset.flatten(),torch.tensor([len(flat)],device=offset.device,dtype=offset.dtype)])
    out=torch.zeros((12,len(uv)),device=uv.device)
    if len(tile):_surface[(len(tile),)](m,c,z,o,source_id,source_local,color_valid,unobserved,off,flat,uv,order,tile,begin,end,target_id,target_local,foot,raw_depth,aligned_depth,out,gaussian_mass,len(uv),num_warps=8)
    return out


@tr.jit
def _delta(M,C,O,OFFSET,IDS,DC,COLOR,OUT,BASE,ALPHA,W:tl.constexpr,H:tl.constexpr,GW:tl.constexpr,ZERO:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0);p=tl.arange(0,256);px=tile%GW*16+p%16;py=tile//GW*16+p//16;good=(px<W)&(py<H)
    T=tl.full((256,),1.,tl.float32);active=good;mass=tl.full((256,),0.,tl.float32);delta=tl.full((8,256),0.,tl.float32);current=tl.full((4,256),0.,tl.float32)
    base=tl.load(OFFSET+tile);end=tl.load(OFFSET+tile+1)
    while (base<end)&(tl.sum(active.to(tl.int32),0)>0):
        j=base+tl.arange(0,B);inside=j<end;idx=tl.load(IDS+j,inside,other=0)
        dx=tl.load(M+2*idx,inside,other=0)[None,:]-px[:,None]-.5;dy=tl.load(M+2*idx+1,inside,other=0)[None,:]-py[:,None]-.5
        c0=tl.load(C+3*idx,inside,other=0);c1=tl.load(C+3*idx+1,inside,other=0);c2=tl.load(C+3*idx+2,inside,other=0)
        sigma=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        a=tl.minimum(.999,tl.load(O+idx,inside,other=0)[None,:]*tl.exp(-sigma));a=tl.where(inside[None,:]&active[:,None]&(sigma>=0)&(a>=1/255),a,0.)
        prefix=tl.cumprod(1-a,1);after=T[:,None]*prefix;weight=tl.where(after>1e-4,T[:,None]*prefix/tl.maximum(1-a,1e-8)*a,0.)
        for channel in tl.static_range(6):
            value=tl.sum(weight*tl.load(DC+6*idx+channel,inside,other=0)[None,:],1);delta+=tl.where(tl.arange(0,8)[:,None]==channel,value[None,:],0.)
        if ZERO:
            for channel in tl.static_range(3):
                value=tl.sum(weight*tl.load(COLOR+3*idx+channel,inside,other=0)[None,:],1);current+=tl.where(tl.arange(0,4)[:,None]==channel,value[None,:],0.)
        mass+=tl.sum(weight,1);T*=tl.sum(tl.where(tl.arange(0,B)[None,:]==B-1,prefix,0.),1);active&=T>1e-4;base+=B
    for channel in tl.static_range(6):
        value=tl.sum(tl.where(tl.arange(0,8)[:,None]==channel,delta,0.),0);tl.store(OUT+(py*W+px)*6+channel,value,good)
    if ZERO:
        for channel in tl.static_range(3):
            value=tl.sum(tl.where(tl.arange(0,4)[:,None]==channel,current,0.),0);tl.store(BASE+(py*W+px)*3+channel,value,good)
    tl.store(ALPHA+py*W+px,mass,good)


@torch.no_grad()
def delta_color_statistics(scene,K,view,delta_colors,width=512,height=512,include_current=False):
    m,z,c,o,offset,flat=projection(scene,K,view,width,height);off=torch.cat([offset.flatten(),torch.tensor([len(flat)],device=K.device,dtype=offset.dtype)])
    out=torch.empty((height,width,6),device=K.device);current=torch.empty((height,width,3),device=K.device);alpha=torch.empty((height,width),device=K.device)
    _delta[(math.ceil(width/16)*math.ceil(height/16),)](m,c,o,off,flat,delta_colors.contiguous(),scene.colors.contiguous(),out,current,alpha,width,height,math.ceil(width/16),include_current,num_warps=8)
    return out,current,alpha
