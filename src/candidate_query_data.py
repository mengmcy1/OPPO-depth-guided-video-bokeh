"""Numpy-only data interface for future candidate decoders; no network."""
import numpy as np


def observation_flat_index(slots,source_xy,width=512,height=512):
    """Global source-reference indices; promote before pixel arithmetic."""
    xy=np.asarray(source_xy,np.int64);s=np.asarray(slots,np.int64)
    if np.any((xy[:,0]<0)|(xy[:,0]>=width)|(xy[:,1]<0)|(xy[:,1]>=height)|(s<0)):raise ValueError('Invalid source observation reference')
    return s*width*height+xy[:,1]*width+xy[:,0]


def pack_queries(pool,band,max_candidates=8,max_queries=32768):
    """Uniform spatial query cap; sort candidate observations by input depth.

    No GT, target hidden texture, scoring region or label is accepted here.
    Padded indices must be masked before gathering candidate observations.
    """
    pixels=np.flatnonzero(band);all_count=len(pixels)
    if all_count>max_queries:pixels=pixels[np.linspace(0,all_count-1,max_queries,dtype=np.int64)]
    order=np.lexsort((pool['depth'],pool['pixel']));sorted_pixel=pool['pixel'][order];start=np.searchsorted(sorted_pixel,pixels,'left');end=np.searchsorted(sorted_pixel,pixels,'right');counts=np.minimum(end-start,max_candidates);indices=np.full((len(pixels),max_candidates),-1,np.int32)
    for column in range(max_candidates):
        valid=counts>column;indices[valid,column]=order[start[valid]+column]
    return dict(query_pixel=pixels.astype(np.int32),query_xy=np.c_[pixels%band.shape[1]+.5,pixels//band.shape[1]+.5].astype(np.float32),candidate_indices=indices,candidate_valid=indices>=0,candidate_count=counts.astype(np.uint8)),dict(full_band_queries=all_count,saved_queries=len(pixels),max_queries=max_queries,M=max_candidates,sort='estimated depth ascending;whole observation retained',cap='deterministic uniform sorted-spatial selection,noGT',second_hidden_slot_reserved=True)


def query_features(pool,indices,valid):
    """Finite input-only tensors, with explicit missing geometric evidence."""
    ids=np.maximum(indices,0);evidence=np.stack([pool[k] for k in ['reprojection','FB','angle_radians']],-1);missing=~np.isfinite(evidence);evidence=np.where(missing,np.array([8.,8.,0.]),evidence)
    numeric=np.c_[pool['position_world_estimated'],pool['depth'],pool['colors'],pool['edge_distance'],evidence,pool['weight'],pool['source_xy'].astype(np.float32)/512,pool['source_frame_id'].astype(np.float32)/60,pool['edge'],missing].astype(np.float32)
    if not np.isfinite(numeric).all():raise ValueError('Nonfinite candidate feature outside declared missing evidence')
    features=np.where(valid[...,None],numeric[ids],0);patches=np.where(valid[...,None,None,None],pool['rgb_patch'][ids],0)
    return dict(features=features,source_RGB_patch=patches,candidate_valid=valid,feature_missing_mask=np.where(valid[...,None],missing[ids],False)),dict(shape=list(features.shape),columns=['world_x','world_y','world_z','target_camera_depth','linear_R','linear_G','linear_B','source_edge_distance_px','reprojection_px','FB_px','parallax_radians','confidence','source_u_div512','source_v_div512','source_frame_div60','source_edge','missing_reprojection','missing_FB','missing_parallax'],missing_evidence_fill=[8.,8.,0.],GT_used=False)
