"""Instrument the frozen Query's actual soft source mixture and vary its RGB only."""
from dataclasses import replace

import numpy as np
import torch

from route3_v2data_E0_train import predict
from route3_v2data_hidden_color_sources import CUTS, OUT, bin_counts
from route3_v2data_stage3b import FrameQueries, optical_band
from src.aperture_renderer import GaussianScene
from src.footprint_order import footprint_scale
from src.ordered_layers import order_hidden


def geometry_cloud(network, store, frame, case, depth, K):
    """Mirror Stage 3b O1 and its footprint mask, retaining survivor provenance."""
    active=store.active("native",case["focus_m"],case["fstop"])
    if not len(active):
        return None,None
    residuals=[]
    sources=[]
    source_colors=network.source_colors
    def take_residual(module,inputs,value):
        residuals.append(network.color_bound*value[...,:3].tanh())
    def trace_sources(world,slots,context):
        rgb,inside=source_colors(world,slots,context)
        pose=context["c2w"][slots]
        camera=torch.einsum("bhmi,bmij->bhmj",world[:,:,None]-pose[:,None,:,:3,3],pose[:,:,:3,:3])
        uv=(camera@context["K"].T)[...,:2]/camera[...,2:].clamp_min(1e-8)
        sources.append((rgb,inside,uv))
        return rgb,inside
    hook=network.appearance.register_forward_hook(take_residual)
    network.source_colors=trace_sources
    row=dict(frame=frame,input="native",case=case)
    try:
        outputs=[store.forward(network,row,part,{"size":depth.shape[-1]})[0]
                 for part in np.array_split(active,(len(active)+2047)//2048)]
    finally:
        del network.source_colors
        hook.remove()
    def join(key):
        return torch.cat([p[key].flatten(0,1) for p in outputs])
    xyz,colors,opacity=map(join,("means_camera","colors","opacities"))
    projected=xyz@K.T
    uv=projected[:,:2]/projected[:,2:].clamp_min(1e-8)
    h,w=depth.shape
    ix=uv[:,0].floor().long().clamp(0,w-1)
    iy=uv[:,1].floor().long().clamp(0,h-1)
    band=optical_band(depth.cpu().numpy(),K,case["focus_m"],case["fstop"])
    inside=((xyz[:,2]>0)&(opacity>1e-4)&(uv>=0).all(-1)&(uv[:,0]<w)&(uv[:,1]<h)
            &torch.as_tensor(band,device=K.device)[iy,ix])
    if not inside.any():
        return None,None
    xyz,colors,opacity=(v[inside] for v in (xyz,colors,opacity))
    sigma=xyz[:,2]/K[0,0]*.7
    quats=xyz.new_zeros((len(xyz),4))
    quats[:,0]=1
    raw=GaussianScene(xyz,quats,torch.stack((sigma,sigma,sigma*.01),-1),opacity,colors,
                      torch.ones_like(opacity,dtype=torch.long))
    ordered,ordering=order_hidden(raw,depth,K,"O1")
    if ordering["dropped"]:
        raise ValueError("The frozen O1 geometry unexpectedly dropped rows")
    factor,keep=footprint_scale(ordered.means,ordered.scales[:,:2],ordered.scales[:,2],depth,K)
    if not keep.any():
        return None,None
    scales=torch.cat((ordered.scales[keep,:2]*factor[keep,None],ordered.scales[keep,2:]),1)
    result=GaussianScene(ordered.means[keep],ordered.quats[keep],scales,ordered.opacities[keep],
                         ordered.colors[keep],ordered.source_frames[keep])
    retained=inside.clone()
    retained[inside]=keep
    ids=active[retained.nonzero(as_tuple=True)[0].cpu().numpy()//2]
    fields=store.features
    slots=torch.as_tensor(fields["source_slots"][ids],dtype=torch.long,device=K.device)
    valid=torch.as_tensor(fields["candidate_valid"][ids],device=K.device)
    weights=join("weights")[retained]
    rgb,in_view,coords=(torch.cat([part[i].flatten(0,1) for part in sources])[retained] for i in range(3))
    effective=weights*in_view*valid
    effective=effective/effective.sum(-1,keepdim=True).clamp_min(1e-8)
    delta=torch.cat(residuals).flatten(0,1)[retained]
    original=((rgb*effective[...,None]).sum(1)+delta).clamp(0,1)
    if not torch.allclose(result.colors,original,atol=2e-6,rtol=0):
        difference=float((result.colors-original).abs().max())
        raise ValueError(f"Frozen Query appearance did not reproduce the actual RGB source mixture: {difference}")
    return result,dict(slots=slots.cpu().numpy(),source_xy=fields["source_xy"][ids],
                       uv=coords.cpu().numpy(),weights=effective,rgb=rgb,delta=delta)


def rgb_coverage(evidence, view, uv, object_id, proxy):
    """Integrate the exact four source pixels used by align_corners=False RGB."""
    shifted=uv-.5
    origin=np.floor(shifted).astype(int)
    frac=shifted-origin
    gt=np.zeros(len(uv),np.float32)
    pred=np.zeros_like(gt)
    for yy in (0,1):
        for xx in (0,1):
            coefficient=(frac[:,0] if xx else 1-frac[:,0])*(frac[:,1] if yy else 1-frac[:,1])
            xy=(origin+[xx,yy]).clip(0,511)
            gt += coefficient*evidence.coverage(view,xy,object_id)
            pred += coefficient*proxy[xy[:,1],xy[:,0]]
    return gt,pred


def read_proxy(evidence, frame, callable):
    path=OUT/"predicted-purity"/evidence.name/f"f{frame:03d}.npy"
    if not path.exists():
        value=callable(frame)
        if value.shape!=(512,512) or not np.isfinite(value).all():
            raise ValueError(f"E-H2 purity map invalid for {evidence.name}/{frame}")
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix(".part").open("wb") as stream:
            np.save(stream,value)
        path.with_suffix(".part").replace(path)
    return np.load(path)


def anchor(evidence, frame, pixels):
    points,back=np.unique(pixels.astype(int),axis=0,return_inverse=True)
    pose=evidence.poses[frame-1]
    directions=np.column_stack((points+.5,np.ones(len(points))))@evidence.Kinv.T@pose[:3,:3].T
    exact,hit=evidence.world.cast(frame,pose[:3,3],directions)
    ids=np.where(hit>=0,evidence.world.tri_obj[hit.clip(min=0)],0)
    return exact[back],ids[back]


def error_by_bin(colors, pure, support, target_rgb):
    values={}
    for i in range(4):
        mask=support&(np.searchsorted(CUTS,pure,side="right")==i)
        err=colors[mask].astype(np.float64)-target_rgb[mask]
        values[str(i)]=dict(samples=int(mask.sum()),absolute=float(np.abs(err).sum()),squared=float(np.square(err).sum()))
    return values


def color_variants(cloud, data, evidence, views, frame, proxy):
    n=len(cloud.means)
    source=np.asarray([a["frame"] for a in views])
    for view in views:
        f=view["frame"]
        if not np.allclose(view["K"],evidence.K,atol=.01) or not np.allclose(view["c2w"],evidence.poses[f-1],atol=1e-4):
            raise ValueError(f"Query source projection and GT pose disagree for {evidence.name}/{f}")
    rgb=data["rgb"].cpu().numpy()
    weights=data["weights"].cpu().numpy()
    slots, uv=data["slots"],data["uv"]
    ar=np.arange(n)
    dominant=np.argmax(weights,axis=1)
    winning_view=source[slots[ar,dominant]]
    world=np.zeros((n,3),np.float64)
    identity=np.zeros(n,np.int16)
    for f in np.unique(winning_view):
        found=winning_view==f
        world[found],identity[found]=anchor(evidence,int(f),data["source_xy"][ar[found],dominant[found]])
    gt=np.full(weights.shape,np.nan,np.float32)
    pred=gt.copy()
    for index in np.unique(slots[weights>0]):
        f=int(source[index])
        r,c=np.nonzero((slots==index)&(weights>0))
        lookup=read_proxy(evidence,f,proxy)
        g,p=rgb_coverage(evidence,f,uv[r,c],identity[r],lookup)
        gt[r,c]=np.where(identity[r]>0,g,np.nan)
        pred[r,c]=p
    strongest=gt[ar,dominant]
    clean,peeled=evidence.peeled(frame,world,identity)
    finite=np.isfinite(gt)
    source_rows,source_cols=np.nonzero(weights>0)
    source_purity=gt[source_rows,source_cols]
    summed=np.bincount(np.searchsorted(CUTS,gt[finite],side="right"),
                       weights=weights[finite],minlength=4).tolist()
    stats=dict(retained_heads=n,dominant_gt=bin_counts(strongest),
               contributing_gt=bin_counts(source_purity),
               weighted_gt=summed,unknown_weight=float(weights[~finite].sum()),
               peeled_matched=int((clean&np.isfinite(strongest)).sum()),
               peeled_color_error=error_by_bin(rgb[ar,dominant],strongest,
                                               clean&np.isfinite(strongest),peeled),
               contributing_color_error=error_by_bin(rgb[source_rows,source_cols],source_purity,
                    clean[source_rows]&np.isfinite(source_purity),peeled[source_rows]))
    modes={"Query":cloud}
    best={}
    for typ,values in (("QGT",gt),("QPred",pred)):
        known=np.isfinite(values)&(weights>0)
        maximum=np.where(known,values,-1).max(1)
        best[typ]=np.where(maximum>=0,maximum,np.nan)
        stats[typ]=dict(best=bin_counts(best[typ]),admitted={})
        for thresh in (0.,*CUTS.tolist()):
            # Only the colour weights have their impure contributions removed.
            # Geometry and opacity are kept from the original head wherever it has colour.
            good=known&(values>=thresh)&(values>0)
            keep=good.any(1)
            w=data["weights"]*torch.as_tensor(good,device=cloud.means.device)
            w=w/w.sum(1,keepdim=True).clamp_min(1e-8)
            replacement=((data["rgb"]*w[...,None]).sum(1)+data["delta"]).clamp(0,1)
            attrs={key:getattr(cloud,key)[keep] for key in cloud.__dataclass_fields__}
            attrs["colors"]=replacement[keep]
            key=f"{typ}@{thresh:.2f}"
            modes[key]=GaussianScene(**attrs) if keep.any() else None
            stats[typ]["admitted"][key]=int(keep.sum())
    return modes,stats,best


def proxy_generator(geometry, e_network, e_inputs, evidence, sample, train_frames):
    def compute(frame):
        x,_=sample(("B",evidence.name,train_frames.index(frame)))
        forecast=geometry(x,protect_residual=True,render_heads=False,return_features=True)
        cached=e_inputs.observations(evidence.name,frame,x,forecast)
        mass=predict(e_network,cached)["log_mass"][0,1].exp()
        return (2*(mass-.5).abs()).cpu().numpy()
    return compute
