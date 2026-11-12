"""Rule-only observation memory. Predictions and labels are never written here."""
import numpy as np


def build_memory(position,color,confidence,source_reference,voxel_m=.02,max_hypotheses=4,verified=None):
    """Conservative spatial/color cells retain CSR links to original observations.

    Up to four color hypotheses per spatial voxel; raw observations remain in
    the caller's archive, including ones pruned from this compact index.
    No GT, neural feature or inferred hidden surface enters this operation.
    """
    if verified is not None:
        selected=np.flatnonzero(verified)
        if not len(selected):raise ValueError('No verified observations for long-term memory')
        memory,stats=build_memory(position[selected],color[selected],confidence[selected],source_reference[selected],voxel_m,max_hypotheses)
        entry=np.full(len(position),-1,np.int32);entry[selected]=memory['observation_entry_id'];memory['observation_entry_id']=entry;memory['observation_indices']=selected[memory['observation_indices']].astype(np.int32);memory['original_source_references']=source_reference
        stats.update(observations=len(position),verified_observations=len(selected),unverified_observations_kept_in_raw_only=int((~verified).sum()),input_only_verification=True)
        return memory,stats
    spatial=np.floor(position/voxel_m).astype(np.int32);color_cell=np.floor(np.clip(color,0,1)*16).astype(np.int16)
    keys=np.concatenate([spatial,color_cell],axis=1);unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    weights=np.maximum(confidence.astype(np.float64),1e-8);sums=np.bincount(inverse,weights=weights,minlength=len(unique))
    voxels,first=np.unique(unique[:,:3],axis=0,return_index=True);ends=np.r_[first[1:],len(unique)];keep=np.zeros(len(unique),bool)
    for start,end in zip(first,ends):keep[start+np.argsort(-sums[start:end],kind='stable')[:max_hypotheses]]=True
    mapping=np.full(len(unique),-1,np.int32);mapping[keep]=np.arange(keep.sum());entry=mapping[inverse];retained=entry>=0;gid=entry[retained];den=np.bincount(gid,weights=weights[retained],minlength=keep.sum())
    mean=np.stack([np.bincount(gid,weights=weights[retained]*position[retained,d],minlength=keep.sum())/den for d in range(3)],-1)
    rgb=np.stack([np.bincount(gid,weights=weights[retained]*color[retained,d],minlength=keep.sum())/den for d in range(3)],-1)
    sigma=np.stack([np.bincount(gid,weights=weights[retained]*(position[retained,d]-mean[gid,d])**2,minlength=keep.sum())/den for d in range(3)],-1)
    order=np.flatnonzero(retained);order=order[np.argsort(entry[order],kind='stable')];count=np.bincount(entry[order],minlength=keep.sum());ptr=np.r_[0,np.cumsum(count)]
    return dict(position=mean.astype(np.float32),covariance_diagonal=sigma.astype(np.float32),color=rgb.astype(np.float16),confidence=den.astype(np.float32),feature=np.empty((len(mean),0),np.float16),observation_entry_id=entry,observation_indices=order.astype(np.int32),observation_ptr=ptr.astype(np.int64),original_source_references=source_reference,spatial_color_key=unique[keep],voxel_m=np.float32(voxel_m)),dict(observations=len(position),entries=len(mean),spatial_voxels=len(voxels),pruned_observations=int((~retained).sum()),max_hypotheses_per_voxel=max_hypotheses,feature_dimension=0,original_observations_preserved=True)
