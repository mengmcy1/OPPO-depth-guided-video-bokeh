"""Separate, label-only two-surface packing and matched-ray objectives.

Depth clustering here only packs GT-positive labels into the two output
slots. Candidate content correctness remains the world-point/ray label.
"""
import numpy as np
import torch
import torch.nn.functional as F


def pack_targets(labels,inputs):
    hit=labels['content_hit_fraction'];valid=inputs['candidate_valid'];correct=(hit>0)&valid;depth=labels['candidate_depth_target'];rgb=labels['candidate_RGB_target'];camera=labels['candidate_camera_target'];K=inputs['target_K'];uvz=camera@K.T
    with np.errstate(invalid='ignore',divide='ignore'):uv=uvz[...,:2]/uvz[...,2:]
    distance=np.nan_to_num(((uv-inputs['query_xy'][:,None])**2).sum(-1),nan=1e9,posinf=1e9);usable=correct&labels['placement_support_v2']&labels['appearance_valid']&np.isfinite(depth)&(depth>0)&np.isfinite(rgb).all(-1)
    b,m=hit.shape;targets=dict(q_target=correct.astype(np.float32),hit_weight=hit.astype(np.float32),depth=np.zeros((b,2),np.float32),color=np.zeros((b,2,3),np.float32),camera=np.zeros((b,2,3),np.float32),valid=np.zeros((b,2),bool),selection=np.zeros((b,2,m),np.float32),ray_weight=np.zeros((b,2),np.float32),hidden_needed=labels['hidden_any_at_query'].astype(bool),recoverable=labels['recoverable_at_query'].astype(bool));rows=np.arange(b)
    for head in range(2):
        first=np.min(np.where(usable,depth,np.inf),axis=-1);first=np.where(np.isfinite(first),first,0);group=usable&(abs(depth-first[:,None])<=np.maximum(.01,.02*first[:,None]));score=np.where(group,hit/(1+distance/9),0);anchor=score.argmax(-1);present=group.any(-1);targets['valid'][:,head]=present;targets['depth'][:,head]=np.where(present,depth[rows,anchor],0);targets['color'][:,head]=np.where(present[:,None],rgb[rows,anchor],0);targets['camera'][:,head]=np.where(present[:,None],camera[rows,anchor],0);targets['selection'][rows,head,anchor]=present;targets['ray_weight'][:,head]=np.where(present,hit[rows,anchor],0);usable&=~group
    return targets


def objectives(pred,target,candidate_valid):
    zero=pred['depth'].sum()*0;valid=target['valid'];weight=target['ray_weight'][valid]
    if bool(valid.any()):
        depth=((pred['depth'][valid]-target['depth'][valid]).abs()/target['depth'][valid].clamp_min(.1)*weight).sum()/weight.sum().clamp_min(1e-8)
        color=((pred['colors'][valid]-target['color'][valid]).abs().mean(-1)*weight).sum()/weight.sum().clamp_min(1e-8)
        selection=-(target['selection'][valid]*F.log_softmax(pred['selection_logits'][valid],-1)).sum(-1).mean()
    else:depth=color=selection=zero
    qloss=F.binary_cross_entropy_with_logits(pred['q_logits'],target['q_target'],reduction='none');positive=candidate_valid&(target['q_target']>.5);negative=candidate_valid&~positive;parts=[]
    if bool(positive.any()):parts.append((qloss[positive]*target['hit_weight'][positive]).sum()/target['hit_weight'][positive].sum().clamp_min(1e-8))
    if bool(negative.any()):parts.append(qloss[negative].mean())
    q=sum(parts)/len(parts) if parts else zero
    opacity=F.binary_cross_entropy_with_logits(pred['opacity_logits'],valid.to(pred['opacity_logits'].dtype)*.999,pos_weight=pred['opacity_logits'].new_tensor(4.))
    total=depth+color+.2*q+.2*selection+.2*opacity
    return total,dict(total=total,relative_depth_L1=depth,linear_RGB_L1=color,q_BCE=q,selection_CE=selection,opacity_BCE=opacity)


def direct_metrics(pred,target,valid,K):
    """Return sufficient statistics, so batches pool by counts rather than dB."""
    with torch.no_grad():
        correct=target['q_target']>.5;guess=pred['q_logits']>0;tp=(guess&correct&valid).sum();fp=(guess&~correct&valid).sum();fn=(~guess&correct&valid).sum();tn=(~guess&~correct&valid).sum();chosen=pred['weights'].argmax(-1);selected=torch.gather(correct[:,None].expand(-1,2,-1),2,chosen[...,None])[...,0];heads=target['valid'];zerr=(pred['depth'][heads]-target['depth'][heads]).abs()/target['depth'][heads].clamp_min(.1);rgb=(pred['colors'][heads]-target['color'][heads]).abs().mean(-1);w=target['ray_weight'][heads]
        alpha=pred['opacities'].max(-1).values>.1;unneeded=~target['hidden_needed'];needed=target['hidden_needed'];p=pred['means_camera']@K.T;q=target['camera']@K.T;uv=p[...,:2]/p[...,2:].clamp_min(1e-8);gtuv=q[...,:2]/q[...,2:].clamp_min(1e-8);xy=(uv[heads]-gtuv[heads]).norm(dim=-1)
        return dict(TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),selected_correct=int((selected&heads).sum()),matched_heads=int(heads.sum()),matched_ray_weight=float(w.sum()),weighted_relative_depth_sum=float((zerr*w).sum()),weighted_linear_RGB_MAE_sum=float((rgb*w).sum()),weighted_placement_pixel_error_sum=float((xy*w).sum()),false_activated_queries=int((alpha&unneeded).sum()),unneeded_queries=int(unneeded.sum()),activated_needed_queries=int((alpha&needed).sum()),needed_queries=int(needed.sum()),candidate_correct_available_queries=int(correct.any(-1).sum()),queries=len(valid))


def summarize(stats):
    n=stats['TP']+stats['FP']+stats['FN']+stats['TN'];p=stats['TP']+stats['FP'];r=stats['TP']+stats['FN'];w=stats['matched_ray_weight']
    return dict(**stats,candidate_accuracy=(stats['TP']+stats['TN'])/max(n,1),candidate_precision=stats['TP']/max(p,1),candidate_recall=stats['TP']/max(r,1),selection_correct_rate=stats['selected_correct']/stats['matched_heads'] if stats['matched_heads'] else None,matched_ray_relative_depth_L1=stats['weighted_relative_depth_sum']/w if w else None,matched_ray_linear_RGB_MAE=stats['weighted_linear_RGB_MAE_sum']/w if w else None,placement_pixel_error=stats['weighted_placement_pixel_error_sum']/w if w else None,false_activation_rate=stats['false_activated_queries']/max(stats['unneeded_queries'],1),needed_activation_recall=stats['activated_needed_queries']/max(stats['needed_queries'],1))
