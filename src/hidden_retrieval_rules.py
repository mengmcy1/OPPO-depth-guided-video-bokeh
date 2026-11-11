"""Provenance-preserving, input-only cumulative hidden retrieval rules."""
import numpy as np


def collect(target_depth,K,target_c2w,band,views,target_frame):
    h,w=target_depth.shape;parts=[]
    for slot,view in enumerate(views):
        if view['frame']==target_frame:continue
        z=view['depth'];good=np.isfinite(z)&(z>0);yy,xx=np.nonzero(good);depth=z[good].astype(np.float64)
        rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(view['K']).T
        world=(rays*depth[:,None])@view['c2w'][:3,:3].T+view['c2w'][:3,3]
        camera=(world-target_c2w[:3,3])@target_c2w[:3,:3];uvz=camera@K.T;uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-10)
        x=np.floor(uv[:,0]).astype(np.int32);y=np.floor(uv[:,1]).astype(np.int32);inside=(camera[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h);ids=np.flatnonzero(inside);ids=ids[band[y[ids],x[ids]]]
        visible=target_depth[y[ids],x[ids]];ids=ids[camera[ids,2]>visible+np.maximum(.001,.005*visible)]
        parts.append(dict(pixel=(y[ids]*w+x[ids]).astype(np.int32),depth=camera[ids,2].astype(np.float32),colors=view['rgb_linear'][yy[ids],xx[ids]],
            slot=np.full(len(ids),slot,np.uint8),source_xy=np.stack([xx[ids],yy[ids]],-1).astype(np.int16),edge=view['edge'][yy[ids],xx[ids]],weight=view['confidence'][yy[ids],xx[ids]]))
    return {key:np.concatenate([p[key] for p in parts]) for key in parts[0]}


def consensus_groups(depth,color,slots,indices):
    """All-pair-consistent greedy groups, one witness per input frame."""
    order=np.argsort(depth,kind='stable');groups=[];seen=set()
    dz=abs(depth[:,None]-depth[None]);tol=np.maximum(.02,.02*np.minimum(depth[:,None],depth[None]));agree=(dz<=tol)&(abs(color[:,None]-color[None]).max(-1)<=.05)
    for seed in order:
        group=[int(seed)];sources={int(slots[seed])};candidates=np.lexsort((indices,abs(depth-depth[seed])))
        for j in candidates:
            if int(slots[j]) in sources or not agree[j,group].all():continue
            group.append(int(j));sources.add(int(slots[j]))
        if len(sources)<2:continue
        key=tuple(sorted(group))
        if key not in seen:groups.append(np.array(group,np.int32));seen.add(key)
    return groups


def aggregate(pool,stage,visible_depth,cluster_cache=None):
    """R0 reproduces the historical nearest-depth/median path exactly."""
    n=len(pool['depth']);keep=np.ones(n,bool);pix=pool['pixel'];y=pix//512;x=pix%512
    if stage>=1:keep&=pool['depth']>visible_depth[y,x]+np.maximum(.005,.02*visible_depth[y,x])
    if stage>=2:keep&=~pool['edge']
    indices=np.flatnonzero(keep);indices=indices[np.lexsort((pool['depth'][indices],pix[indices]))]
    if stage<=2 and len(indices):
        p,first=np.unique(pix[indices],return_index=True);group=np.cumsum(np.r_[True,np.diff(pix[indices])!=0])-1;nearest=pool['depth'][indices[first]]
        indices=indices[pool['depth'][indices]<=nearest[group]*1.01]
    pixels,first,counts=np.unique(pix[indices],return_index=True,return_counts=True)
    output={k:[] for k in ['uv','depth','colors_linear','source_mask','source_count','contrib_start','contrib_count']};contributors=[]
    for pixel,start,count in zip(pixels,first,counts):
        group=indices[start:start+count]
        if stage>=3:
            if cluster_cache is not None and int(pixel) in cluster_cache:clusters=cluster_cache[int(pixel)]
            else:
                local=consensus_groups(pool['depth'][group],pool['colors'][group],pool['slot'][group],group);clusters=[group[c] for c in local]
                if cluster_cache is not None:cluster_cache[int(pixel)]=clusters
            if not clusters:continue
            if stage==3:group=clusters[0]
            else:group=max(clusters,key=lambda ids:(float(pool['weight'][ids].sum()),-float(np.median(pool['depth'][ids]))))
        slots=np.unique(pool['slot'][group]);output['uv'].append([pixel%512+.5,pixel//512+.5]);output['contrib_start'].append(len(contributors));output['contrib_count'].append(len(group));contributors.extend(group.tolist())
        if stage==4:
            weight=pool['weight'][group].astype(np.float64);weight/=max(weight.sum(),1e-12);output['depth'].append(float(weight@pool['depth'][group]));output['colors_linear'].append(weight@pool['colors'][group])
        else:output['depth'].append(np.median(pool['depth'][group]));output['colors_linear'].append(np.median(pool['colors'][group],axis=0))
        output['source_mask'].append(sum(1<<int(s) for s in slots));output['source_count'].append(len(slots))
    types=dict(uv=np.float32,depth=np.float32,colors_linear=np.float32,source_mask=np.uint16,source_count=np.uint8,contrib_start=np.int64,contrib_count=np.int32)
    result={k:np.asarray(v,dtype=types[k]) for k,v in output.items()};result['uv']=result['uv'].reshape(-1,2);result['colors_linear']=result['colors_linear'].reshape(-1,3);result['contrib_indices']=np.array(contributors,np.int64)
    return result,dict(pool_candidates=n,after_prefilter=int(keep.sum()),parents=len(result['depth']),stage=stage)
