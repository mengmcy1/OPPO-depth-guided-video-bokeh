"""Derive resolution/context-specific input-only queries and separate ray labels.

CPU entry in the existing dofgs environment (Embree). Geometry/recoverability
are recast for the five-frame context, never resized from a 13-frame mask.
No Blender render or trainable model is used here.
"""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt,uniform_filter
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def save_npz(path,value):
    temporary=path.with_name(path.name+'.part')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**value)
    temporary.replace(path)

def save_json(path,value):
    temporary=path.with_name(path.name+'.part');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

from src.hidden_retrieval_rules import collect
from src.candidate_query_data import pack_queries,query_features,observation_flat_index
from src.dual_layer import depth_band
ALL=[1,6,11,16,21,26,30,35,40,45,50,55,60];FIVE=[1,16,30,45,60]
OUT=ROOT/'outputs/route3-network-s2a-2026-11-12-01';PILOT=ROOT/'outputs/route3-network-pilot-2026-11-12-01'

def calibration_applied(row):
    fit=row['models']['SHARP']
    if 'calibration_applied' in fit:return bool(fit['calibration_applied'])
    return bool(fit.get('success',False) and fit.get('heldout_not_worse',False) and not fit.get('identity_fallback',False) and not fit.get('fit_rejected_and_native_depth_retained',False))


def resize(x,size,linear=False):
    if x.shape[:2]==(size,size):return x.copy()
    if linear:return cv2.resize(x,(size,size),interpolation=cv2.INTER_AREA)
    factor=512//size;return x[factor-1::factor,factor-1::factor].copy()

def descriptor(name):
    if name=='B2-high':
        b=ROOT/'outputs/route3-b2only-2026-11-10-01';c=ROOT/'outputs/route3-sharp-s1c-2026-11-10-01';e=[x for x in json.loads((c/'prepared.json').read_text())['rows'] if x['geometry']=='cal_E1'];s=ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high';return dict(scene=s,entries=e,geometry=b/'B2-high/geometry',model=ROOT/'outputs/route3-sharp-s1-2026-11-11-01/depth',calibration=c/'calibration/E1/SHARP',confidence=ROOT/'outputs/route3-sharp-s1d-2026-11-11-01/confidence',parents=lambda f:b/'methods-3d/prepared'/f'B2-high-f{f:03d}'/'parents.npz',source_root=None)
    s=OUT/name
    if not s.exists():s=PILOT/name
    return dict(scene=s,entries=json.loads((s/'prepared.json').read_text())['rows'],geometry=s/'geometry',model=s/'models/SHARP',calibration=s/'models/calibration/SHARP',confidence=s/'models/calibration',parents=lambda f:s/'oracle'/f'f{f:03d}.npz',source_root=s)

def views_for(desc,frames,size,GT=False,world=None):
    views=[];K=np.array([[size*50/36,0,size/2],[0,size*50/36,size/2],[0,0,1]],np.float32)
    for f in frames:
        m=np.load(desc['model']/f'f{f:03d}.npz');c=np.load(desc['calibration']/f'f{f:03d}.npz');confpath=desc['confidence']/f'f{f:03d}-confidence.npz'
        if not confpath.exists():confpath=desc['confidence']/f'f{f:03d}.npz'
        conf=np.load(confpath);z=resize(c['depth'],size);rgb=resize(m['rgb_linear'],size,True);inv=1/np.maximum(z,1e-6);hi=cv2.dilate(inv,np.ones((3,3),np.uint8));lo=cv2.erode(inv,np.ones((3,3),np.uint8));width=max(1,int(round(3*size/512)));edge=cv2.dilate(((hi-lo)>.03*hi).astype(np.uint8),np.ones((2*width+1,2*width+1),np.uint8))>0
        if GT:z=world.center_full(f,1)['depth_axial_m'];z=np.where(z>0,z,1e4)
        raw=dict(frame=f,depth=z,K=K,c2w=c['c2w'],rgb_linear=rgb,rgb_srgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055).clip(0,1).astype(np.float32),edge=edge,edge_distance=distance_transform_edt(~edge).astype(np.float32),confidence=resize(conf['confidence' if 'confidence' in conf.files else 'weight'],size))
        for k in ['reprojection','FB','angle_radians']:raw[k]=resize(conf[k],size)*(size/512 if k in ['reprojection','FB'] else 1)
        views.append(raw)
    return views

def placement_map(desc,entry):
    s=desc['source_root'];f=entry['frame']
    if s is not None:
        lab=np.load(s/'labels'/f'f{f:03d}'/(entry['cases'][0]['id']+'.npz'));v=np.load(s/'labels'/f'f{f:03d}'/'placement-support-v2.npz');return lab['placement_affine_a'],lab['placement_affine_b'],v['local_fit_supported']
    g=np.load(desc['geometry']/f'f{f:03d}-center.npz');z=np.load(entry['input_npz'])['depth'];gt=g['depth_axial_m'];fg=np.load(desc['scene']/'scene-geometry.npz')['foreground_object_ids'];valid=g['center_visible']&~np.isin(g['object_id'],fg);yy,xx=np.mgrid[:512,:512];train=valid&((xx//8+yy//8)%5!=0);x=1/np.maximum(gt,1e-6);y=1/np.maximum(z,1e-6);box=lambda a:uniform_filter(a.astype(np.float64),49,mode='constant')*49**2;n=box(train);sx=box(x*train);sy=box(y*train);var=box(x*x*train)-sx*sx/np.maximum(n,1);cov=box(x*y*train)-sx*sy/np.maximum(n,1);a=np.clip(np.where(var>1e-5,cov/np.maximum(var,1e-12),1),.1,3);b=(sy-a*sx)/np.maximum(n,1);valid=n>=32
    if valid.any():ix=distance_transform_edt(~valid,return_distances=False,return_indices=True);a=np.where(valid,a,a[tuple(ix)]);b=np.where(valid,b,b[tuple(ix)])
    return a.astype(np.float32),b.astype(np.float32),valid


def input_cache_512(name,desc,out):
    """512/13 inference consumes no hidden GT, labels or evaluation masks."""
    views=views_for(desc,ALL,512);context=out/'context.npz'
    if not context.exists():save_npz(context,dict(frames=np.array(ALL),images=np.stack([v['rgb_linear'] for v in views]),depth=np.stack([v['depth'] for v in views]),K=views[0]['K'],c2w=np.stack([v['c2w'] for v in views])))
    rows=[]
    for entry in desc['entries']:
        f=entry['frame'];target=next(v for v in views if v['frame']==f);band=np.zeros((512,512),bool)
        for c in entry['cases']:band|=depth_band(target['depth'],target['K'][0,0],c['focus_m'],.05/(2*c['fstop']))[0]
        rawpool=collect(target['depth'],target['K'],target['c2w'],band,views,f);n=len(rawpool['depth']);xyz=np.empty((n,3),np.float32);patch=np.empty((n,3,3,3),np.uint8);extras={k:np.empty(n,np.float32) for k in ['edge_distance','reprojection','FB','angle_radians']}
        for slot,v in enumerate(views):
            at=rawpool['slot']==slot;x,y=rawpool['source_xy'][at].T;z=v['depth'][y,x];r=np.c_[x+.5,y+.5,np.ones(len(x))]@np.linalg.inv(v['K']).T;xyz[at]=(r*z[:,None])@v['c2w'][:3,:3].T+v['c2w'][:3,3]
            for k in extras:extras[k][at]=v[k][y,x]
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:patch[at,dy+1,dx+1]=np.uint8(np.rint(v['rgb_srgb'][np.clip(y+dy,0,511),np.clip(x+dx,0,511)]*255))
        rawpool.update(position_world_estimated=xyz,source_frame_id=np.array(ALL)[rawpool['slot']],rgb_patch=patch,**extras);q,qa=pack_queries(rawpool,band,8,32768);valid=q['candidate_valid'];ix=np.maximum(q['candidate_indices'],0);gather=lambda x:np.where(valid.reshape(*valid.shape,*([1]*(x.ndim-1))),x[ix],0) if n else np.zeros((*valid.shape,*x.shape[1:]),x.dtype);features=query_features(rawpool,q['candidate_indices'],valid)[0]['features'] if n else np.zeros((*valid.shape,19),np.float32);inp=dict(**q,features=features,position_world=gather(xyz),source_slots=gather(rawpool['slot']),source_xy=gather(rawpool['source_xy']),target_K=target['K'],target_c2w=target['c2w'],target_depth=target['depth'],band=band);source_depth=np.zeros(valid.shape,np.float32)
        for slot,v in enumerate(views):
            at=valid&(inp['source_slots']==slot);xy=inp['source_xy'][at];source_depth[at]=v['depth'][xy[:,1],xy[:,0]]
        inp['source_depth']=source_depth;inp['memory_verified']=np.zeros(valid.shape,bool);inp['memory_entry_id']=np.full(valid.shape,-1,np.int32);s=desc['source_root']
        if s is not None:
            status=np.load(s/'memory/verification.npz')['verified'];entries=np.load(s/'memory/verified-index.npz')['observation_entry_id'];idx=observation_flat_index(rawpool['slot'],rawpool['source_xy']);inp['memory_verified']=gather(status[idx]);inp['memory_entry_id']=gather(entries[idx]);cal=json.loads((s/'models/calibration/summary.json').read_text())['rows'];inp['calibration_applied']=np.array([calibration_applied(next(x for x in cal if x['frame']==frame)) for frame in ALL]);inp['target_calibration_applied']=np.array(calibration_applied(next(x for x in cal if x['frame']==f)))
        else:inp['calibration_applied']=np.ones(13,bool);inp['target_calibration_applied']=np.array(True)
        local=out/f'f{f:03d}';local.mkdir(exist_ok=True);path=local/'inputs.npz'
        if not path.exists():save_npz(path,inp)
        for c in entry['cases']:rows.append(dict(frame=f,case=c,input=str(path),target=None,query_stats=qa))
    save_json(out/'manifest.json',dict(scene=name,rows=rows,context=str(context),size=512,frames=ALL,M=8,features=19,query_cap=32768,model_input_only=True,hidden_GT_not_read=True,calibration_and_memory_flags_loaded=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--size',type=int,choices=[256,512],default=256);p.add_argument('--GT-warmup',action='store_true');a=p.parse_args();cv2.setNumThreads(2);desc=descriptor(a.scene);frames=FIVE if a.size==256 else ALL;size=a.size;tag=f'{size}-{len(frames)}'+('-GTwarmup' if a.GT_warmup else '');out=OUT/'learning-data'/a.scene/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return
    if size==512:
        if a.GT_warmup:raise ValueError('GT warmup only at256')
        input_cache_512(a.scene,desc,out);return
    from route3_blender_b2_full_geometry import FullWorld
    w=FullWorld(desc['scene']/'scene-geometry.npz',size);w.d={k:v for k,v in w.d.items()};w.d['input_frames']=np.array(frames);views=views_for(desc,frames,size,a.GT_warmup,w);native={f:w.center_full(f,1) for f in frames};world={f:w.world(g['local_surface_point'].reshape(-1,3),g['object_id'].reshape(-1),f).reshape(size,size,3).astype(np.float32) for f,g in native.items()};footprints={}
    for f in frames:
        delta=[]
        for axis in [0,1]:
            v=np.linalg.norm(np.diff(world[f],axis=axis),axis=-1);v=np.where(np.diff(native[f]['object_id'],axis=axis)==0,v,0);pad=[(0,0),(0,0)];pad[axis]=(0,1);delta.append(np.pad(v,pad))
        footprints[f]=np.maximum(np.maximum(*delta),1e-4)
    camera_path=out/'context.npz'
    if not camera_path.exists():save_npz(camera_path,dict(frames=np.array(frames),images=np.stack([v['rgb_linear'] for v in views]),depth=np.stack([v['depth'] for v in views]),K=views[0]['K'],c2w=np.stack([v['c2w'] for v in views])))
    rows=[]
    for entry in desc['entries']:
        f=entry['frame'];target=next(v for v in views if v['frame']==f);K=target['K'];pose=target['c2w'];cases=[dict(c) for c in entry['cases']];plan=desc['geometry']/'training-focus-plan.json'
        if plan.exists():cases+=next(r['cases'] for r in json.loads(plan.read_text())['rows'] if r['frame']==f)
        elif size==256:
            z=native[f]['depth_axial_m'][native[f]['center_visible']];low,high=z.min(),z.max()
            for u in (np.arange(4)+.5)/4:
                focus=float(1/((1-u)/low+u/high));cases.append(dict(id=f'f{f:03d}-N1.4-F{focus:.6f}',frame=f,nominal_focus_m=focus,focus_m=focus,fstop=1.4,training_focus_only=True))
        band=np.zeros((size,size),bool)
        for case in cases:band|=depth_band(target['depth'],K[0,0],case['focus_m'],.05/(2*case['fstop']))[0]
        pool=collect(target['depth'],K,pose,band,views,f);n=len(pool['depth']);xyz=np.empty((n,3),np.float32);source=np.empty((n,3),np.float32);oid=np.zeros(n,np.int16);foot=np.zeros(n,np.float32);patch=np.empty((n,3,3,3),np.uint8);extras={k:np.empty(n,np.float32) for k in ['edge_distance','reprojection','FB','angle_radians']};verified=np.zeros(n,bool);entryid=np.full(n,-1,np.int32)
        mem=None;ids=None
        if desc['source_root'] is not None:
            mem=np.load(desc['source_root']/'memory/verification.npz')['verified'];ids=np.load(desc['source_root']/'memory/verified-index.npz')['observation_entry_id']
        for slot,v in enumerate(views):
            take=pool['slot']==slot;sx,sy=pool['source_xy'][take].T;z=v['depth'][sy,sx];ray=np.c_[sx+.5,sy+.5,np.ones(len(sx))]@np.linalg.inv(K).T;xyz[take]=(ray*z[:,None])@v['c2w'][:3,:3].T+v['c2w'][:3,3];source[take]=world[v['frame']][sy,sx];oid[take]=native[v['frame']]['object_id'][sy,sx];foot[take]=footprints[v['frame']][sy,sx]
            for key in extras:extras[key][take]=v[key][sy,sx]
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:patch[take,dy+1,dx+1]=np.uint8(np.rint(v['rgb_srgb'][np.clip(sy+dy,0,size-1),np.clip(sx+dx,0,size-1)]*255))
            if mem is not None:
                xy=np.floor((pool['source_xy'][take].astype(np.float32)+.5)*512/size).astype(int);rawidx=observation_flat_index(np.full(len(xy),ALL.index(v['frame'])),xy);verified[take]=mem[rawidx];entryid[take]=ids[rawidx]
        pool.update(position_world_estimated=xyz,source_frame_id=np.array(frames)[pool['slot']],rgb_patch=patch,**extras);q,qa=pack_queries(pool,band,8,32768)
        if n:feats=query_features(pool,q['candidate_indices'],q['candidate_valid'])[0]['features'];feats[...,12:14]*=512/size
        else:feats=np.zeros((*q['candidate_indices'].shape,19),np.float32)
        ix=np.maximum(q['candidate_indices'],0);valid=q['candidate_valid'];local=out/f'f{f:03d}';local.mkdir(exist_ok=True);inputpath=local/'inputs.npz';safe_gather=lambda x:np.where(valid.reshape(*valid.shape,*([1]*(x.ndim-1))),x[ix],0) if n else np.zeros((*valid.shape,*x.shape[1:]),x.dtype)
        # Public inputs and quality flags: no GT world points, masks or labels.
        inp=dict(**q,features=feats,position_world=safe_gather(xyz),source_slots=safe_gather(pool['slot']),source_xy=safe_gather(pool['source_xy']),source_depth=np.zeros(valid.shape,np.float32),memory_verified=safe_gather(verified),memory_entry_id=safe_gather(entryid),target_K=K,target_c2w=pose,target_depth=target['depth'],band=band)
        for slot,v in enumerate(views):
            take=valid&(inp['source_slots']==slot);sx=inp['source_xy'][...,0];sy=inp['source_xy'][...,1];inp['source_depth'][take]=v['depth'][sy[take],sx[take]]
        if desc['source_root'] is not None:
            cal=json.loads((desc['source_root']/'models/calibration/summary.json').read_text())['rows'];inp['calibration_applied']=np.array([calibration_applied(next(r for r in cal if r['frame']==g)) for g in frames]);inp['target_calibration_applied']=np.array(calibration_applied(next(r for r in cal if r['frame']==f)))
        else:inp['calibration_applied']=np.ones(len(frames),bool);inp['target_calibration_applied']=np.array(True);inp['memory_verified']=np.zeros(valid.shape,bool)
        if not inputpath.exists():save_npz(inputpath,inp)
        gtcam=(source-pose[:3,3])@pose[:3,:3];gtuvz=gtcam@K.T;gtuv=gtuvz[:,:2]/np.maximum(gtuvz[:,2:],1e-8);cx=np.clip(gtuv[:,0].astype(int),0,size-1);cy=np.clip(gtuv[:,1].astype(int),0,size-1);inside=(gtcam[:,2]>0)&(gtuv[:,0]>=0)&(gtuv[:,0]<size)&(gtuv[:,1]>=0)&(gtuv[:,1]<size);center_hidden=inside&(oid>0)&(gtcam[:,2]>native[f]['depth_axial_m'][cy,cx]+1e-4)&(np.linalg.norm(source-world[f][cy,cx],axis=-1)>1.5*foot+1e-4)
        aa,bb,support=placement_map(desc,entry)
        if a.GT_warmup:aa=np.ones_like(aa);bb=np.zeros_like(bb);support=np.ones_like(support)
        sx=np.clip(np.floor(gtuv[:,0]*512/size).astype(int),0,511);sy=np.clip(np.floor(gtuv[:,1]*512/size).astype(int),0,511);mapped=1/np.maximum(aa[sy,sx]/np.maximum(gtcam[:,2],1e-6)+bb[sy,sx],1e-4);mappedcam=gtcam*(mapped/np.maximum(gtcam[:,2],1e-6))[:,None];safe=support[sy,sx]&inside&(oid>0)&(mapped>target['depth'][cy,cx]+np.maximum(.005,.02*target['depth'][cy,cx]))
        oracle=np.load(desc['parents'](f));opc=np.c_[oracle['uv'],np.ones(len(oracle['uv']))]@np.linalg.inv(np.array([[711.111111,0,256],[0,711.111111,256],[0,0,1]])).T*oracle['depth'][:,None];opw=opc@pose[:3,:3].T+pose[:3,3];color=np.zeros((n,3),np.float32);colorvalid=np.zeros(n,bool)
        for obj in np.unique(oid):
            at=np.flatnonzero(oid==obj);parents=np.flatnonzero(oracle['object_id']==obj)
            if obj<=0 or not len(parents):continue
            dist,nearest=cKDTree(opw[parents]).query(source[at]);color[at]=oracle['colors_linear'][parents[nearest]];colorvalid[at]=dist<=3*gtcam[at,2]/K[0,0]+1e-4
        for case in cases:
            key=case['id'];targetpath=local/(key+'-labels.npz')
            if targetpath.exists():rows.append(dict(frame=f,case=case,input=str(inputpath),target=str(targetpath)));continue
            geo,info=w.aperture_full(f,16,case['fstop'],case['nominal_focus_m']);hitworld=w.world(geo['local_surface_point'].reshape(-1,3),geo['object_id'].reshape(-1),f).reshape(16,size,size,3);bits=np.zeros(n,np.uint16)
            for k,xy in enumerate(geo['aperture_xy']):
                cam=gtcam-np.r_[xy,0];intr=K.copy();intr[:2,2]+=np.diag(K)[:2]*xy/case['nominal_focus_m'];p=cam@intr.T;uv=p[:,:2]/np.maximum(p[:,2:],1e-8);x=np.clip(uv[:,0].astype(int),0,size-1);y=np.clip(uv[:,1].astype(int),0,size-1);ok=center_hidden&(uv[:,0]>=0)&(uv[:,0]<size)&(uv[:,1]>=0)&(uv[:,1]<size)&(cam[:,2]>0)&geo['hidden'][k,y,x]&(oid==geo['object_id'][k,y,x])&(np.linalg.norm(source-hitworld[k,y,x],axis=-1)<=1.5*foot+1e-4)
                for slot,v in enumerate(views):
                    at=np.flatnonzero(ok&(pool['slot']==slot));cam_s=(hitworld[k,y[at],x[at]]-v['c2w'][:3,3])@v['c2w'][:3,:3];p=cam_s@K.T;uv_s=p[:,:2]/np.maximum(p[:,2:],1e-8);ok[at]&=(abs(uv_s-(pool['source_xy'][at]+.5))<=.75).all(-1)
                bits|=ok.astype(np.uint16)*(1<<k)
            fraction=np.unpackbits(bits.view(np.uint8).reshape(-1,2),axis=1).sum(1).astype(np.float32)/16;hit=safe_gather(fraction);content=valid&(hit>0);placement_valid=safe_gather(safe)&valid;rgb_valid=safe_gather(colorvalid)&valid;label=dict(content_hit_fraction=hit,content_hit_bits=safe_gather(bits),candidate_depth_target=safe_gather(mapped),candidate_camera_target=safe_gather(mappedcam.astype(np.float32)),candidate_RGB_target=safe_gather(color),placement_support_v2=placement_valid,appearance_valid=rgb_valid,hidden_any_at_query=geo['hidden'].any(0).ravel()[q['query_pixel']],recoverable_at_query=geo['mask_recoverable_total'].ravel()[q['query_pixel']],recoverable_mask=geo['mask_recoverable_total'],residual_pixels=geo['residual_pixels'],hidden_ray_mask=geo['hidden'],recoverable_ray_mask=geo['recoverable'])
            save_npz(targetpath,label);row=dict(frame=f,case=case,input=str(inputpath),target=str(targetpath),queries=len(valid),positive_candidates=int(content.sum()),positive_query_fraction=float(content.any(-1).mean()),safe_positive_fraction=float((content&placement_valid).sum()/max(content.sum(),1)),GT_recoverable_pixels=int(geo['mask_recoverable_total'].sum()),context_frames=frames);rows.append(row);save_json(out/'progress.json',dict(rows=rows));print('S2A_DATA',a.scene,tag,f,key,row['positive_query_fraction'],flush=True)
    save_json(out/'manifest.json',dict(scene=a.scene,rows=rows,context=str(camera_path),size=size,frames=frames,M=8,features=19,query_cap=32768,geometry_frontend_cached_from_13_frames=True,candidate_and_RGB_context_only=frames,GT_input_only_during_warmup=a.GT_warmup,five_frame_recoverability_recomputed=True,depth_loss_requires_placement_support_v2=True,calibration_and_memory_flags_loaded=True,input_label_files_separate=True,model_not_implemented_here=True))

if __name__=='__main__':main()
