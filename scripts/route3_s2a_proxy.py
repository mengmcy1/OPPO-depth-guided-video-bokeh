"""Method-independent scene screening; never reads a DOF truth or score.

Geometry is mesh K16, color difficulty compares the actual input-only LaMa
fill lookup with the peeled hidden reference along recoverable rays.
"""
import argparse,json,os
from pathlib import Path
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/route3-network-s2a-2026-11-12-01'
PILOT=ROOT/'outputs/route3-network-pilot-2026-11-12-01'
RECORDS=ROOT/'03_实验记录/records'

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def display(x):return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055).clip(0,1)
def read(p):
    x=cv2.imread(str(p),-1)
    if x is None:raise FileNotFoundError(p)
    return x[...,:3][...,::-1]
def camera_points(uv,z,K):return np.concatenate([uv,np.ones((*uv.shape[:-1],1))],-1)@np.linalg.inv(K).T*z[...,None]

def proxy(geo_path,entry,parent_path=None,refs=None):
    a=np.load(geo_path);d=np.load(entry['input_npz']);fill=np.load(entry['fill'])['fill_rgb_linear'];h,w=d['depth'].shape;K=a['K'];rec=a['recoverable']&a['hidden']&~a['residual_pixels'][None];uv=a['reference_uv_center'];dep=a['depth_axial_m'];obj=a['object_id'];gt=np.zeros((*dep.shape,3),np.float32);good=np.zeros(dep.shape,bool)
    if parent_path:
        p=np.load(parent_path);pc=camera_points(p['uv'],p['depth'],K)
        for oid in np.unique(obj[rec]):
            ps=np.flatnonzero(p['object_id']==oid);ks,ys,xs=np.nonzero(rec&(obj==oid))
            if not len(ps):continue
            xyz=camera_points(uv[ks,ys,xs],dep[ks,ys,xs],K);dist,ix=cKDTree(pc[ps]).query(xyz);ok=dist<=3*dep[ks,ys,xs]/K[0,0]+1e-4;gt[ks,ys,xs]=p['colors_linear'][ps[ix]];good[ks,ys,xs]=ok
    else:
        p1=a['peel_first_excluded_id'];p2=a['peel_second_excluded_id'];count=a['peel_count'];xf=uv[...,0]-.5;yf=uv[...,1]-.5
        # Each surviving reference keeps all non-excluded indirect illumination.
        for j in refs:
            ids=j['hidden_object_ids'];mask=rec&(np.isin(p1,ids))&(count==len(ids))
            if len(ids)==2:mask&=np.isin(p2,ids)
            if not mask.any():continue
            rgb=read(j['output']).clip(0,1)
            for k in range(16):
                take=mask[k];v=cv2.remap(rgb,xf[k].astype(np.float32),yf[k].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);gt[k][take]=v[take];good[k]|=take
    rec_pixels=a['mask_recoverable_total'];valid=rec&good;yy,xx=np.mgrid[:h,:w];case=next(c for c in entry['cases'] if abs(c['nominal_focus_m']-float(a['focus_m']))<1e-6) if 'focus_m' in a.files else None
    # Caller sets the nominal group; method focus follows the frozen target-click rule.
    nominal=float(entry['proxy_nominal_focus_m']);focus=next(c['focus_m'] for c in entry['cases'] if abs(c['nominal_focus_m']-nominal)<1e-6 and abs(c['fstop']-1.4)<1e-6)
    bg=d['background_depth'];sse=0.;n=0;pixel_sse=np.zeros((h,w),np.float64);pixel_count=np.zeros((h,w),np.int32)
    for k,xy in enumerate(a['aperture_xy']):
        shift=xy[None,None,:]*np.array([K[0,0],K[1,1]])*(1/np.maximum(bg,1e-4)-1/focus)[...,None];u=(xx+shift[...,0]).astype(np.float32);v=(yy+shift[...,1]).astype(np.float32);col=cv2.remap(fill,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);err=((display(col)-display(gt[k]))**2).sum(-1);ok=valid[k];pixel_sse+=np.where(ok,err,0);pixel_count+=ok
    have=pixel_count>0;mean=pixel_sse[have]/pixel_count[have];mse=float(mean.mean()/3) if len(mean) else None
    return dict(frame=int(a['frame_id']),focus_m=nominal,fstop=1.4,recoverable_pixels=int(rec_pixels.sum()),recoverable_image_fraction=float(rec_pixels.mean()),recoverable_rays=int(rec.sum()),color_supported_rays=int(valid.sum()),color_support_fraction=float(valid.sum()/max(rec.sum(),1)),fill_hidden_PSNR_dB=float(-10*np.log10(max(mse,1e-12))) if mse is not None else None,color_scored_pixels=int(have.sum()),color_SSE=float(mean.sum()) if len(mean) else 0.,mesh_geometry=str(geo_path),fill=entry['fill'],actual_method_focus_m=focus)

def known():
    rows=[];b2=ROOT/'outputs/route3-b2only-2026-11-10-01';s1c=ROOT/'outputs/route3-sharp-s1c-2026-11-10-01';entries=[e for e in json.loads((s1c/'prepared.json').read_text())['rows'] if e['geometry']=='cal_E1']
    for label in ['B2-high','pilot-thin','pilot-multilayer']:
        es=entries if label=='B2-high' else json.loads((PILOT/label/'prepared.json').read_text())['rows'];values=[]
        for e in es:
            f=e['frame'];e=dict(e)
            for focus in [1.4,5.]:
                e['proxy_nominal_focus_m']=focus;geo=(b2/'B2-high/geometry' if label=='B2-high' else PILOT/label/'geometry')/f'f{f:03d}-N1.4-F{focus:g}-K16.npz';parent=b2/'methods-3d/prepared'/f'B2-high-f{f:03d}'/'parents.npz' if label=='B2-high' else PILOT/label/'oracle'/f'f{f:03d}.npz';values.append(proxy(geo,e,parent_path=parent))
        rows.append(dict(scene=label,rows=values))
    write(OUT/'screen-calibration-proxies.json',dict(rows=rows,DOF_images_or_method_scores_read=False,colors_from_existing_peeled_reference=True,definition='pixel-weighted display-sRGB fill error across GT recoverable K16 rays;actual input-side LaMa+extendeddepth lookup;GT masks only proxy,not a method'))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--known',action='store_true');a=p.parse_args()
    if a.known:known()
