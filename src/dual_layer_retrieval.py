"""Input-only geometric hidden-layer retrieval; no target hidden GT accepted."""
import numpy as np


def retrieve(target_depth,target_K,target_c2w,band,views,target_frame,
             depth_margin_relative=.005,aggregation_relative=.01):
    """Nearest behind-visible layer, median RGB/depth within1% of its depth.

    Every view contains only frame, current visible depth/K/pose and RGB.
    Moving surfaces are not compensated, and are diagnosed using GT later.
    """
    h,w=target_depth.shape;collected=[];audit=[]
    for slot,view in enumerate(views):
        if view['frame']==target_frame:continue
        z=view['depth'];good=np.isfinite(z)&(z>0)
        yy,xx=np.nonzero(good);depth=z[good].astype(np.float64)
        rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(view['K']).T
        world=(rays*depth[:,None])@view['c2w'][:3,:3].T+view['c2w'][:3,3]
        camera=(world-target_c2w[:3,3])@target_c2w[:3,:3];uvz=camera@target_K.T
        uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-10);x=np.floor(uv[:,0]).astype(np.int32);y=np.floor(uv[:,1]).astype(np.int32)
        inside=(camera[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h);ids=np.flatnonzero(inside)
        inband=band[y[ids],x[ids]];ids=ids[inband]
        visible=target_depth[y[ids],x[ids]];margin=np.maximum(.001,depth_margin_relative*visible)
        behind=camera[ids,2]>visible+margin;accepted=ids[behind]
        audit.append(dict(source_frame=int(view['frame']),source_visible_pixels=len(depth),inside_target_pixels=int(inside.sum()),
            projected_in_band=len(ids),rejected_not_behind_visible=int((~behind).sum()),accepted=int(behind.sum())))
        if len(accepted):
            collected.append(dict(pixel=y[accepted]*w+x[accepted],depth=camera[accepted,2].astype(np.float32),
                color=view['rgb_linear'][yy[accepted],xx[accepted]],source_slot=np.full(len(accepted),slot,np.int16)))
    if not collected:
        return dict(uv=np.empty((0,2),np.float32),depth=np.empty(0,np.float32),colors_linear=np.empty((0,3),np.float32),
                    source_mask=np.empty(0,np.uint16),source_count=np.empty(0,np.uint8)),audit
    data={k:np.concatenate([q[k] for q in collected]) for k in collected[0]}
    order=np.lexsort((data['depth'],data['pixel']));data={k:v[order] for k,v in data.items()}
    pix,first=np.unique(data['pixel'],return_index=True);group=np.cumsum(np.r_[True,np.diff(data['pixel'])!=0])-1
    nearest=data['depth'][first];keep=data['depth']<=nearest[group]*(1+aggregation_relative)
    data={k:v[keep] for k,v in data.items()};pix,first,counts=np.unique(data['pixel'],return_index=True,return_counts=True)
    result_depth=[];colors=[];source_bits=[];source_count=[]
    for start,count in zip(first,counts):
        section=slice(start,start+count);slots=np.unique(data['source_slot'][section])
        result_depth.append(np.median(data['depth'][section]));colors.append(np.median(data['color'][section],axis=0))
        source_bits.append(sum(1<<int(k) for k in slots));source_count.append(len(slots))
    parent=dict(uv=np.stack([pix%w+.5,pix//w+.5],-1).astype(np.float32),depth=np.array(result_depth,np.float32),
        colors_linear=np.array(colors,np.float32),source_mask=np.array(source_bits,np.uint16),source_count=np.array(source_count,np.uint8))
    target_slot=next(i for i,v in enumerate(views) if v['frame']==target_frame)
    if np.any(parent['source_mask']&(1<<target_slot)):raise ValueError('Current target leaked into cross-frame retrieval')
    return parent,audit
