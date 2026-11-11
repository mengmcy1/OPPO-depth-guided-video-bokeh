"""S1b video-only sparse geometry and affine inverse-depth calibration.

This executable reads input RGB, estimated depth, calibration and the given
E1/E2 poses. It does not load GT depth, semantic IDs, hidden surfaces or DOF.
"""
import datetime
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
S1=ROOT/'outputs/route3-sharp-s1-2026-11-11-01'
UPPER=ROOT/'outputs/route3-upper-bound-2026-11-11-01'
OUT=ROOT/'outputs/route3-sharp-s1b-2026-11-11-01'
RECORDS=ROOT/'03_实验记录/records';PREFIX='route3-sharp-s1b-2026-11-11'
CRITERIA=RECORDS/'route3-sharp-s1b-criteria-2026-11-11.json'
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def bilinear(image,xy):
    x=np.clip(xy[:,0],0,image.shape[1]-1);y=np.clip(xy[:,1],0,image.shape[0]-1);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,image.shape[1]-1);y1=np.minimum(y0+1,image.shape[0]-1);dx=x-x0;dy=y-y0
    return (1-dx)*(1-dy)*image[y0,x0]+dx*(1-dy)*image[y0,x1]+(1-dx)*dy*image[y1,x0]+dx*dy*image[y1,x1]


def fundamental(K,cam_i,cam_j):
    a=np.linalg.inv(cam_i);b=np.linalg.inv(cam_j);rel=b@cam_i;t=rel[:3,3]
    skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0.]])
    return np.linalg.inv(K).T@skew@rel[:3,:3]@np.linalg.inv(K)


def observations(features,edges,poses,K):
    offsets=np.cumsum([0]+[len(x['xy']) for x in features]);total=offsets[-1]
    parent=np.arange(total);bits=np.repeat(1<<np.arange(len(features)),np.diff(offsets));rank=np.zeros(total,np.uint8)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    accepted=[];pair_stats=[]
    for i,j,links in edges:
        if np.linalg.norm(poses[i,:3,3]-poses[j,:3,3])<.01:continue
        F=fundamental(K,poses[i],poses[j]);good=[]
        for qi,tj,distance in links:
            x=np.r_[features[i]['xy'][qi]+.5,1.];y=np.r_[features[j]['xy'][tj]+.5,1.]
            l=F@x;m=F.T@y;error=abs(float(y@l))
            if max(error/max(np.linalg.norm(l[:2]),1e-12),error/max(np.linalg.norm(m[:2]),1e-12))<=1.:
                good.append((distance,offsets[i]+qi,offsets[j]+tj))
        accepted+=good;pair_stats.append(dict(frame_i=FRAMES[i],frame_j=FRAMES[j],descriptor_matches=len(links),epipolar_matches=len(good)))
    for _,a,b in sorted(accepted):
        a,b=find(a),find(b)
        if a==b or bits[a]&bits[b]:continue
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a;bits[a]|=bits[b]
        if rank[a]==rank[b]:rank[a]+=1
    groups={}
    for i,f in enumerate(features):
        for kp in range(len(f['xy'])):
            root=find(offsets[i]+kp)
            if int(bits[root]).bit_count()>=3:groups.setdefault(root,[]).append((i,kp))
    extrinsics=np.linalg.inv(poses);P=np.einsum('ij,njk->nik',K,extrinsics[:,:3,:]);tracks=[]
    for nodes in groups.values():
        A=[]
        for i,kp in nodes:
            x,y=features[i]['xy'][kp]+.5;A.extend([x*P[i,2]-P[i,0],y*P[i,2]-P[i,1]])
        _,_,vh=np.linalg.svd(np.stack(A));hom=vh[-1]
        if abs(hom[3])<1e-10:continue
        world=hom[:3]/hom[3];errors=[];depths=[];rays=[]
        for i,kp in nodes:
            camera=extrinsics[i,:3,:3]@world+extrinsics[i,:3,3];depths.append(camera[2]);q=K@camera
            errors.append(np.linalg.norm(q[:2]/max(q[2],1e-12)-(features[i]['xy'][kp]+.5)))
            ray=world-poses[i,:3,3];rays.append(ray/max(np.linalg.norm(ray),1e-12))
        if min(depths)<=0 or max(errors)>1.:continue
        rays=np.stack(rays);angle=float(np.rad2deg(np.arccos(np.clip((rays@rays.T).min(),-1,1))))
        if angle<.1:continue
        tracks.append(dict(world=world,nodes=nodes,depths=depths,reprojection=max(errors),angle_deg=angle))
    return tracks,pair_stats


def fit_inverse(predicted,triangulated,full_depth):
    x=1/np.asarray(predicted,np.float64);y=1/np.asarray(triangulated,np.float64);xmin=float(1/np.max(full_depth));epsilon=1e-4
    def objective(v):
        residual=v[0]*x+v[1]-y;delta=.005
        h=np.where(abs(residual)<=delta,.5*residual**2,delta*(abs(residual)-.5*delta))
        return float(h.mean()+1e-6*((v[0]-1)**2+v[1]**2))
    result=minimize(objective,[1.,0.],method='SLSQP',bounds=[(.05,20.),(-2.,2.)],
        constraints=[dict(type='ineq',fun=lambda v:v[0]*xmin+v[1]-epsilon)],options=dict(maxiter=500,ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all():raise RuntimeError('S1B_STOP inverse calibration numeric fit: '+result.message)
    return dict(a=float(result.x[0]),b=float(result.x[1]),minimum_predicted_inverse_depth=xmin,objective=float(result.fun),iterations=int(result.nit),success=True)


def inverse_stats(pred,tri,a=1.,b=0.):
    inv=a/np.asarray(pred,np.float64)+b;truth=1/np.asarray(tri,np.float64)
    return dict(points=len(truth),inverse_RMSE=float(np.sqrt(np.mean((inv-truth)**2))),inverse_MAE=float(np.mean(abs(inv-truth))),
        sparse_depth_AbsRel=float(np.mean(abs(1/np.maximum(inv,1e-4)-tri)/tri)))


def main():
    c=json.loads(CRITERIA.read_text());assert c['frozen'];cv2.setNumThreads(8);start=time.monotonic()
    folder=OUT/'calibration';folder.mkdir(parents=True,exist_ok=False);features=[];pose_by={};depths={}
    for frame in FRAMES:
        d=np.load(S1/'depth'/f'f{frame:03d}.npz');gray=cv2.cvtColor(np.uint8(np.rint(d['rgb_srgb']*255)),cv2.COLOR_RGB2GRAY)
        detector=cv2.SIFT_create(nfeatures=4000,contrastThreshold=.02,edgeThreshold=10,sigma=1.6);kp,desc=detector.detectAndCompute(gray,None)
        features.append(dict(xy=np.array([p.pt for p in kp],np.float64),desc=desc));depths[frame]=dict(SHARP=d['depth'],DA3=np.load(UPPER/'U2-da3/E1'/f'f{frame:03d}.npz')['depth'])
    for variant in ['E1','E2']:pose_by[variant]=np.stack([np.load(UPPER/'U2-da3'/variant/f'f{frame:03d}.npz')['c2w'] for frame in FRAMES])
    K=d['K'].astype(np.float64);matcher=cv2.BFMatcher(cv2.NORM_L2);edges=[]
    for i in range(len(FRAMES)):
        for j in range(i+1,len(FRAMES)):
            ij=matcher.knnMatch(features[i]['desc'],features[j]['desc'],k=2);ji=matcher.knnMatch(features[j]['desc'],features[i]['desc'],k=2)
            reverse={q.queryIdx:q.trainIdx for q,z in ji if q.distance<.75*z.distance}
            links=[(q.queryIdx,q.trainIdx,float(q.distance)) for q,z in ij if q.distance<.75*z.distance and reverse.get(q.trainIdx)==q.queryIdx]
            edges.append((i,j,links))
        print('CALIB_FEATURE_PAIRS',FRAMES[i],flush=True)
    results={};stop=None
    for variant_index,variant in enumerate(['E1','E2']):
        tracks,pair_stats=observations(features,edges,pose_by[variant],K);rng=np.random.default_rng(2026111122+variant_index)
        order=rng.permutation(len(tracks));validation=set(order[int(.8*len(order)):].tolist());rows=[]
        for i,frame in enumerate(FRAMES):
            valid=[(track_id,t,t['nodes'].index(next(node for node in t['nodes'] if node[0]==i))) for track_id,t in enumerate(tracks) if any(node[0]==i for node in t['nodes'])]
            xy=np.array([features[i]['xy'][t['nodes'][k][1]] for _,t,k in valid],np.float64).reshape(-1,2)
            tri=np.array([t['depths'][k] for _,t,k in valid]);isheld=np.array([tid in validation for tid,_,_ in valid],dtype=bool);train=~isheld
            cells=set((np.floor(xy[train,1]/128)*4+np.floor(xy[train,0]/128)).astype(int).tolist())
            span=float(np.quantile(1/tri[train],.9)-np.quantile(1/tri[train],.1)) if train.any() else 0.
            supported=int(train.sum())>=100 and int(isheld.sum())>=25 and len(cells)>=6 and span>=.03
            row=dict(frame=frame,points=len(valid),train_points=int(train.sum()),heldout_points=int(isheld.sum()),occupied_train_cells=len(cells),
                train_inverse_depth_span_P90_P10=span,near_points_below_2_5m=int((tri<2.5).sum()),far_points_atleast_2_5m=int((tri>=2.5).sum()),supported=supported,
                triangulated_depth_p10_p50_p90=np.quantile(tri,[.1,.5,.9]).tolist() if len(tri) else None,models={})
            np.savez_compressed(folder/f'{variant}-f{frame:03d}-sparse.npz',xy=xy,depth=tri,heldout=isheld)
            if not supported:
                stop=f'{variant} frame{frame}: train{train.sum()} held{isheld.sum()} cells{len(cells)} inverse-span{span:.5f} below frozen support';rows.append(row);break
            for source,z in depths[frame].items():
                observed=bilinear(z,xy);fitted=fit_inverse(observed[train],tri[train],z)
                mapped=1/np.maximum(fitted['a']/z+fitted['b'],1e-4)
                dest=folder/variant/source;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'f{frame:03d}.npz',depth=mapped.astype(np.float32),K=K,c2w=pose_by[variant][i])
                row['models'][source]=dict(**fitted,train_before=inverse_stats(observed[train],tri[train]),train_after=inverse_stats(observed[train],tri[train],fitted['a'],fitted['b']),
                    heldout_before=inverse_stats(observed[isheld],tri[isheld]),heldout_after=inverse_stats(observed[isheld],tri[isheld],fitted['a'],fitted['b']),
                    floored_depth_pixels=int((fitted['a']/z+fitted['b']<1e-4).sum()),output=str(dest/f'f{frame:03d}.npz'))
            rows.append(row);print('CALIB_FRAME',variant,frame,row['points'],row['models']['SHARP']['a'],row['models']['SHARP']['b'],flush=True)
        results[variant]=dict(tracks=len(tracks),pair_stats=pair_stats,rows=rows)
        if stop:break
    record=dict(completed=stop is None,stop_reason=stop,variants=results,feature_counts=[dict(frame=f,count=len(v['xy'])) for f,v in zip(FRAMES,features)],
        seconds=time.monotonic()-start,GT_depth_or_segmentation_read=False,GT_poses_explicit_E1=True,estimated_depths_and_input_RGB_only=True,criteria=str(CRITERIA))
    write(folder/'result.json',record);write(RECORDS/(PREFIX+'-calibration.json'),record)
    if stop:
        if not (OUT/'stop.json').exists():write(OUT/'stop.json',dict(stage='calibration',reason=stop,record=str(RECORDS/(PREFIX+'-calibration.json'))))
        raise RuntimeError('S1B_STOP '+stop)


if __name__=='__main__':main()
