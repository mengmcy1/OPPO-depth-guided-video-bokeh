"""Moment-matched CoC expansion for a fixed, untrained Gaussian cloud.

This is an inference prototype: the Triton compositor has no backward. The
existing differentiable sampled renderer is unchanged. A lower alpha cutoff
is necessary after area-normalized expansion of dense subpixel Gaussians;
the ordinary gsplat1/255 cutoff would remove most broad, weak components.
"""
import math
import torch
import triton as tr
import triton.language as tl
from .aperture_renderer import aperture_disk
from .layered_aperture import aperture_fill_samples


@tr.jit
def _raster(M,C,F,O,OFFSET,IDS,OUT,ALPHA,W:tl.constexpr,H:tl.constexpr,
            GW:tl.constexpr,B:tl.constexpr=32):
    tile=tl.program_id(0);p=tl.arange(0,256)
    px=(tile%GW)*16+p%16;py=(tile//GW)*16+p//16
    valid=(px<W)&(py<H);start=tl.load(OFFSET+tile)
    # Last offset is a separately appended sentinel, not a Gaussian ID.
    end=tl.load(OFFSET+tile+1)
    T=tl.full((256,),1.,tl.float32);r=tl.full((256,),0.,tl.float32)
    g=r;b=r;d=r
    base=start
    while (base<end)&(tl.max(tl.where(valid,T,0.),0)>=1e-4):
        a=base+tl.arange(0,B);inside=a<end
        idx=tl.load(IDS+a,inside,other=0)
        mx=tl.load(M+2*idx,inside,other=0);my=tl.load(M+2*idx+1,inside,other=0)
        c0=tl.load(C+3*idx,inside,other=0);c1=tl.load(C+3*idx+1,inside,other=0);c2=tl.load(C+3*idx+2,inside,other=0)
        opacity=tl.load(O+idx,inside,other=0)
        dx=mx[None,:]-px[:,None]-.5;dy=my[None,:]-py[:,None]-.5
        sigma=.5*(c0[None,:]*dx*dx+c2[None,:]*dy*dy)+c1[None,:]*dx*dy
        alpha=tl.minimum(.999,opacity[None,:]*tl.exp(-sigma))
        alpha=tl.where((sigma>=0)&(alpha>=1e-6)&inside[None,:]&valid[:,None],alpha,0.)
        attenuation=1-alpha
        prefix=tl.cumprod(attenuation,1)
        weight=prefix/attenuation*alpha
        rr=tl.load(F+4*idx,inside,other=0);gg=tl.load(F+4*idx+1,inside,other=0)
        bb=tl.load(F+4*idx+2,inside,other=0);dd=tl.load(F+4*idx+3,inside,other=0)
        r+=T*tl.sum(weight*rr[None,:],1);g+=T*tl.sum(weight*gg[None,:],1)
        b+=T*tl.sum(weight*bb[None,:],1);d+=T*tl.sum(weight*dd[None,:],1)
        T*=tl.sum(tl.where(tl.arange(0,B)[None,:]==B-1,prefix,0.),1)
        base+=B
    off=(py*W+px)*4
    tl.store(OUT+off,r,valid);tl.store(OUT+off+1,g,valid);tl.store(OUT+off+2,b,valid);tl.store(OUT+off+3,d,valid)
    tl.store(ALPHA+py*W+px,1-T,valid)


def expanded_projection(scene,K,E,width,height,focus,aperture,quadrant=None,eps2d=.0015):
    from gsplat.cuda._wrapper import fully_fused_projection
    # Projection must not cull a point before its expanded CoC reaches the
    # frame. All authorized B2 CoCs fit this64px conservative projection pad.
    pad=64;project_K=K.clone();project_K[:2,2]+=pad
    radii,means,z,conics,AA=fully_fused_projection(scene.means,None,scene.quats,scene.scales,E[None],project_K[None],width+2*pad,height+2*pad,
        eps2d=eps2d,near_plane=1e-4,far_plane=1e4,calc_compensations=True)
    means=means-pad
    c=conics[0];depth=z[0];detinv=(c[:,0]*c[:,2]-c[:,1].square()).clamp_min(1e-20)
    det=detinv.reciprocal();xx=c[:,2]*det;xy=-c[:,1]*det;yy=c[:,0]*det
    if torch.is_tensor(focus) and focus.ndim:
        grid=torch.stack([2*means[0,:,0]/width-1,2*means[0,:,1]/height-1],-1)[None,None]
        focus=torch.nn.functional.grid_sample(focus[None,None],grid,align_corners=False,padding_mode='border')[0,0,0]
    inv=depth.clamp_min(1e-8).reciprocal()-1/focus
    rx=K[0,0]*aperture*inv;ry=K[1,1]*aperture*inv
    if quadrant is None:
        vx=rx.square()/4;vy=ry.square()/4;vxy=torch.zeros_like(rx)
    else:
        m=4/(3*math.pi);sx,sy=quadrant
        vx=rx.square()*(.25-m*m);vy=ry.square()*(.25-m*m)
        vxy=rx*ry*sx*sy*(1/(2*math.pi)-m*m)
    nx,ny,nxy=xx+vx,yy+vy,xy+vxy;newdet=(nx*ny-nxy.square()).clamp_min(1e-20)
    expanded=torch.stack([ny/newdet,-nxy/newdet,nx/newdet],-1)
    opacity=scene.opacities*AA[0]*(det/newdet).clamp_min(0).sqrt()
    newrad=torch.ceil(3.33*torch.stack([nx.sqrt(),ny.sqrt()],-1)).int()
    newrad[(radii[0]<=0).any(-1)]=0
    return means,z,expanded[None].contiguous(),opacity[None].contiguous(),newrad[None],dict(
        below_native_alpha_cutoff_fraction=float((opacity<1/255).float().mean()),
        below_prototype_alpha_cutoff_fraction=float((opacity<1e-6).float().mean()))


def analytic_single(scene,K,E,width,height,focus,aperture,quadrant=None,eps2d=.0015):
    from gsplat.cuda._wrapper import isect_tiles,isect_offset_encode
    means,z,conics,opacities,radii,stats=expanded_projection(scene,K,E,width,height,focus,aperture,quadrant,eps2d)
    gw,gh=math.ceil(width/16),math.ceil(height/16)
    _,intersections,flatten=isect_tiles(means,radii,z,16,gw,gh)
    offsets=isect_offset_encode(intersections,1,gw,gh).reshape(-1)
    offsets=torch.cat([offsets,torch.tensor([len(flatten)],device=offsets.device,dtype=offsets.dtype)])
    features=torch.cat([scene.colors,z[0,:,None]],-1).contiguous()
    image=torch.empty(height,width,4,device=scene.means.device);alpha=torch.empty(height,width,device=image.device)
    _raster[(gw*gh,)](means.contiguous(),conics,features,opacities,offsets,flatten,image,alpha,width,height,gw,num_warps=8)
    return dict(rgb_linear=image[...,:3],alpha=alpha,depth=image[...,3]/alpha.clamp_min(1e-10),**stats)


def sampled(scene,K,E,width,height,focus,aperture,samples,fill,bgdepth,random_seed=None,eps2d=.0015,far_plane=1e4,return_fill_contribution=False):
    from gsplat import rasterization
    fills,_,_=aperture_fill_samples(fill,bgdepth,K,focus,aperture,samples)
    if random_seed is None:disk=aperture_disk(samples,aperture,device=K.device,dtype=K.dtype)
    else:
        generator=torch.Generator(device=K.device);generator.manual_seed(random_seed)
        u=torch.rand(samples,2,device=K.device,generator=generator);theta=2*math.pi*u[:,1]
        disk=aperture*torch.stack((u[:,0].sqrt()*theta.cos(),u[:,0].sqrt()*theta.sin()),-1)
        # Fill lookups must follow these same random rays, not the fixed disk.
        h,w=bgdepth.shape;y,x=torch.meshgrid(torch.arange(h,device=K.device),torch.arange(w,device=K.device),indexing='ij')
        z=torch.where(bgdepth>0,bgdepth,torch.full_like(bgdepth,1e4));uv=torch.stack([x+.5,y+.5],-1)[None]+disk[:,None,None]*torch.stack([K[0,0],K[1,1]])*(1/z-1/focus)[None,...,None]
        grid=torch.stack([2*uv[...,0]/w-1,2*uv[...,1]/h-1],-1)
        fills=torch.nn.functional.grid_sample(fill.permute(2,0,1)[None].expand(samples,3,h,w),grid,align_corners=False,padding_mode='border').permute(0,2,3,1)
    rgb=torch.zeros_like(fill);alpha=torch.zeros_like(bgdepth)
    fill_contribution=torch.zeros_like(fill) if return_fill_contribution else None
    for i,xy in enumerate(disk):
        view=E.clone();view[:2,3]-=xy
        intr=K.clone();intr[0,2]+=K[0,0]*xy[0]/focus;intr[1,2]+=K[1,1]*xy[1]/focus
        image,a,_=rasterization(scene.means,scene.quats,scene.scales,scene.opacities,scene.colors,view[None],intr[None],width,height,
            packed=False,render_mode='RGB+ED',rasterize_mode='antialiased',near_plane=1e-4,far_plane=far_plane,eps2d=eps2d)
        a=a[0,...,0];rgb+=image[0,...,:3]+(1-a[...,None])*fills[i];alpha+=a
        if return_fill_contribution:fill_contribution+=(1-a[...,None])*fills[i]
    result=dict(rgb_linear=rgb/samples,alpha=alpha/samples)
    if return_fill_contribution:result['fill_contribution_linear']=fill_contribution/samples
    return result


def analytic(scene,K,E,width,height,focus,aperture,samples,fill,bgdepth,eps2d=.0015):
    if samples not in [1,4]:raise ValueError('Analytic mode supports1 or4 nonduplicated quadrant moments')
    if torch.is_tensor(focus) and focus.ndim:
        if focus.shape!=(height,width) or not bool(torch.isfinite(focus).all()) or bool((focus<=0).any()):raise ValueError('Positive finite HxW focal field required')
        if samples!=1:raise ValueError('Quadrant K4 diagnosis currently supports scalar focus only')
        field=focus.to(device=K.device,dtype=K.dtype)
        result=analytic_single(scene,K,E,width,height,field,aperture,None,eps2d)
        return dict(rgb_linear=result['rgb_linear']+(1-result['alpha'][...,None])*fill,alpha=result['alpha'],
            component_statistics=[{k:result[k] for k in ['below_native_alpha_cutoff_fraction','below_prototype_alpha_cutoff_fraction']}])
    # At full-disk K1 mean lens point=0. Quadrants have true area centroids.
    m=4/(3*math.pi);quadrants=[None] if samples==1 else [(1,1),(-1,1),(-1,-1),(1,-1)]
    # Residual fill is evaluated at the same quadrant centers. The residual
    # aperture average of Cfill itself is not modeled, and remains a limitation.
    rgb=torch.zeros_like(fill);alpha=torch.zeros_like(bgdepth);statistics=[]
    for q in quadrants:
        xy=torch.zeros(2,device=K.device) if q is None else torch.tensor(q,device=K.device)*aperture*m
        view=E.clone();view[:2,3]-=xy;intr=K.clone();intr[0,2]+=K[0,0]*xy[0]/focus;intr[1,2]+=K[1,1]*xy[1]/focus
        result=analytic_single(scene,intr,view,width,height,focus,aperture,q,eps2d)
        h,w=bgdepth.shape;y,x=torch.meshgrid(torch.arange(h,device=K.device),torch.arange(w,device=K.device),indexing='ij')
        z=torch.where(bgdepth>0,bgdepth,torch.full_like(bgdepth,1e4));uv=torch.stack([x+.5,y+.5],-1)+xy*torch.stack([K[0,0],K[1,1]])*(1/z-1/focus)[...,None]
        grid=torch.stack([2*uv[...,0]/w-1,2*uv[...,1]/h-1],-1)[None]
        lookup=torch.nn.functional.grid_sample(fill.permute(2,0,1)[None],grid,align_corners=False,padding_mode='border')[0].permute(1,2,0)
        rgb+=result['rgb_linear']+(1-result['alpha'][...,None])*lookup;alpha+=result['alpha']
        statistics.append({k:result[k] for k in ['below_native_alpha_cutoff_fraction','below_prototype_alpha_cutoff_fraction']})
    return dict(rgb_linear=rgb/samples,alpha=alpha/samples,component_statistics=statistics)
