"""Download a selected official Sintel subset via validated HTTP ranges and verify native labels.
No model loading, resizing, label normalization or training. Raw selected members retain
archive CRCs; derived inverse-depth files explicitly carry numeric-only validity.
"""
import requests,io,zipfile,json,collections,hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
class RangeReader(io.RawIOBase):
 def __init__(self,url,size,block=2**20):self.url=url;self.size=size;self.pos=0;self.block=block;self.cache=collections.OrderedDict();self.session=requests.Session();self.received=0;self.disk=Path(__file__).resolve().parents[1]/'data/archives/ours-geometry-pilot-20261022/ranges'/hashlib.sha256(url.encode()).hexdigest()[:12];self.disk.mkdir(parents=True,exist_ok=True)
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('negative position')
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  stop=min(self.pos+n,self.size);parts=[]
  while self.pos<stop:
   start=self.pos//self.block*self.block;end=min(start+self.block,self.size)-1
   if start not in self.cache:
    data,received=self.fetch_block(start)
    self.cache[start]=data;self.received+=received
    while len(self.cache)>4:self.cache.popitem(last=False)
   chunk=self.cache[start];offset=self.pos-start;take=min(stop-self.pos,len(chunk)-offset);parts.append(chunk[offset:offset+take]);self.pos+=take
  return b''.join(parts)
 def fetch_block(self,start):
  end=min(start+self.block,self.size)-1;p=self.disk/f'{start}-{end}.bin'
  if p.exists():
   data=p.read_bytes();assert len(data)==end-start+1;return data,0
  for attempt in range(3):
   try:
    r=requests.get(self.url+f'?range_start={start}&range_end={end}',headers={'Range':f'bytes={start}-{end}'},timeout=(10,40))
    r.raise_for_status();assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
    data=r.content;assert len(data)==end-start+1
    tmp=p.with_suffix('.part');tmp.write_bytes(data);tmp.replace(p);return data,len(data)
   except requests.RequestException:
    if attempt==2:raise
 def prefetch(self,infos):
  blocks=set()
  for i in infos:
   # Header/name/extra overhead is covered conservatively; ZipFile validates CRC later.
   a=i.header_offset//self.block*self.block;b=min(self.size-1,i.header_offset+i.compress_size+len(i.filename.encode('utf-8'))+65566)//self.block*self.block
   blocks.update(range(a,b+1,self.block))
  with ThreadPoolExecutor(max_workers=8) as pool:
   jobs={pool.submit(self.fetch_block,x):x for x in sorted(blocks)}
   for n,f in enumerate(as_completed(jobs),1):
    _,received=f.result();self.received+=received
    if n%20==0 or n==len(jobs):print('RANGES',n,'/',len(jobs),'MiB',round(self.received/2**20,1),flush=True)

import argparse,struct,zlib,time
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/ours-geometry-data-pilot-20261022.json'

def write_json(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 content=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 if p.exists():
  assert p.read_text()==content, f'Preserve differing {p}'
 else:p.write_text(content)

def depth_read(path):
 b=path.read_bytes();assert len(b)>=12 and b[:4]==b'PIEH',f'Bad depth header: {path}'
 w,h=struct.unpack('<ii',b[4:12]);assert 0<w<20000 and 0<h<20000 and len(b)==12+w*h*4
 return np.frombuffer(b,dtype='<f4',offset=12).reshape(h,w).copy()

def camera_read(path):
 b=path.read_bytes();assert len(b)==172 and b[:4]==b'PIEH',f'Bad camera header: {path}'
 a=np.frombuffer(b,dtype='<f8',offset=4);K=a[:9].reshape(3,3).copy();E=a[9:].reshape(3,4).copy()
 assert np.isfinite(a).all() and K[0,0]>0 and K[1,1]>0
 assert np.allclose(E[:,:3]@E[:,:3].T,np.eye(3),atol=1e-4) and abs(np.linalg.det(E[:,:3])-1)<1e-4
 return K,E

def members(config,which):
 paths=[]
 for s in config['sequences']:
  for t in s['frames']:
   prefix=f"{s['sequence']}/frame_{t:04d}"
   paths += [f'training/clean/{prefix}.png'] if which=='images' else [f'training/depth/{prefix}.dpt',f'training/camdata_left/{prefix}.cam']
 paths += ['README.txt'] if which=='images' else ['README_depth.txt','sdk/python/sintel_io.py']
 return paths

def download(config):
 raw=ROOT/config['raw_root'];raw.mkdir(parents=True,exist_ok=True)
 index={};total_received=0
 for which,spec in config['archives'].items():
  rr=RangeReader(spec['url'],spec['bytes']);z=zipfile.ZipFile(rr)
  wanted=sorted(members(config,which),key=lambda n:z.getinfo(n).header_offset)
  rr.prefetch([z.getinfo(n) for n in wanted if not (raw/n).exists()])
  for i,n in enumerate(wanted):
   info=z.getinfo(n);dest=raw/n
   if dest.exists():
    data=dest.read_bytes();assert len(data)==info.file_size and zlib.crc32(data)==info.CRC,f'Existing raw mismatch {dest}'
   else:
    for attempt in range(3):
     try:data=z.read(n);break
     except (requests.RequestException,zipfile.BadZipFile) as error:
      print('RETRY',which,n,attempt,type(error).__name__,flush=True)
      if attempt==2:raise
      time.sleep(1)
    assert len(data)==info.file_size and zlib.crc32(data)==info.CRC
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.part');tmp.write_bytes(data);tmp.replace(dest)
   index[n]=dict(archive=which,bytes=info.file_size,crc32=info.CRC)
   if i%25==0 or i==len(wanted)-1:print('DOWNLOAD',which,i+1,'/',len(wanted),'received_MiB',round(rr.received/2**20,1),flush=True)
  total_received+=rr.received
  rr.session.close()
 write_json(raw/'selected-members.json',index)
 print('DOWNLOAD_COMPLETE',len(index),'members','network_MiB_this_run',round(total_received/2**20,2),flush=True)

def verify(config):
 raw=ROOT/config['raw_root'];derived=ROOT/config['derived_root'];derived.mkdir(parents=True,exist_ok=True)
 index=json.loads((raw/'selected-members.json').read_text())
 for name,entry in index.items():
  p=raw/name;b=p.read_bytes();assert len(b)==entry['bytes'] and zlib.crc32(b)==entry['crc32']
 # Compare depth/camera readers against the unchanged official Python SDK without running its CLI.
 import importlib.util
 spec=importlib.util.spec_from_file_location('sintel_sdk',raw/'sdk/python/sintel_io.py');sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
 rows=[];summaries=[];family_split={};seq_inverse_ranges={}
 for seq in config['sequences']:
  family=seq['source_family'];split=seq['split'];assert family not in family_split or family_split[family]==split;family_split[family]=split
  assert seq['frames']==list(range(1,26))
  stats=[];quantiles=[]
  for t in seq['frames']:
   stem=f"{seq['sequence']}/frame_{t:04d}";rgbp=raw/f'training/clean/{stem}.png';dp=raw/f'training/depth/{stem}.dpt';cp=raw/f'training/camdata_left/{stem}.cam'
   with Image.open(rgbp) as im:im.load();assert im.mode=='RGB';size=im.size
   depth=depth_read(dp);K,E=camera_read(cp);assert depth.shape==(size[1],size[0])
   assert np.array_equal(depth,sdk.depth_read(str(dp)),equal_nan=True)
   k0,e0=sdk.cam_read(str(cp));assert np.array_equal(K,k0) and np.array_equal(E,e0)
   valid=np.isfinite(depth)&(depth>0);assert valid.any()
   inv=np.zeros_like(depth);np.divide(1.,depth,out=inv,where=valid);assert np.isfinite(inv[valid]).all()
   output=derived/'reference'/seq['sequence']/f'frame_{t:04d}.npz'
   if output.exists():
    with np.load(output) as a:assert np.array_equal(a['inverse_depth_m_inv'],inv) and np.array_equal(a['numeric_valid'],valid)
   else:output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,inverse_depth_m_inv=inv,numeric_valid=valid,intrinsics=K,world_to_camera=E)
   row=dict(dataset='MPI-Sintel',official_split='training',pilot_split=split,source_family=family,sequence=seq['sequence'],frame=t,width=size[0],height=size[1],rgb=str(rgbp.relative_to(ROOT)),depth_m=str(dp.relative_to(ROOT)),camera=str(cp.relative_to(ROOT)),derived_reference=str(output.relative_to(ROOT)),numeric_valid_fraction=float(valid.mean()),depth_min_m=float(depth[valid].min()),depth_median_m=float(np.median(depth[valid])),depth_max_m=float(depth[valid].max()),vda_prediction=None,focus=None,K_bokeh=None,annotation_reliability='numeric validity only; boundary/transparency visibility not adjudicated')
   rows.append(row);stats.append(row);quantiles.extend(np.quantile(inv[valid],[.02,.98]).tolist())
  seq_inverse_ranges[seq['sequence']]=[min(quantiles),max(quantiles)]
  summaries.append(dict(sequence=seq['sequence'],source_family=family,pilot_split=split,frames=len(stats),size=list(size),numeric_valid_fraction_min=min(x['numeric_valid_fraction'] for x in stats),depth_min_m=min(x['depth_min_m'] for x in stats),depth_max_m=max(x['depth_max_m'] for x in stats)))
  print('VERIFIED',seq['sequence'],len(stats),size,flush=True)
 assert len(rows)==150 and len(family_split)==6
 assert set(x['source_family'] for x in rows if x['pilot_split']=='pilot_train').isdisjoint(x['source_family'] for x in rows if x['pilot_split']=='pilot_dev')
 manifest=dict(config=config,frames=rows)
 write_json(derived/'manifest.json',manifest)
 write_json(ROOT/(config['record_prefix']+'-manifest.json'),manifest)
 report=dict(status='RGB_DEPTH_CAMERA_PILOT_READY_VDA_AND_TARGET_COORDINATES_PENDING',date='2026-10-22',sequences=summaries,total_frames=len(rows),pilot_train_frames=100,pilot_dev_frames=50,source_family_overlap=0,official_SDK_depth_camera_equal=True,selected_raw_members_CRC_verified=len(index),derived_numeric_validity_only=True,raw_bytes=sum(x['bytes'] for x in index.values()),derived_reference_bytes=sum(p.stat().st_size for p in (derived/'reference').rglob('*.npz')),model_inference_runs=0,training_runs=0,limitations=[config['evaluation_scope'],config['validity_policy'],'Native aspect ratio retained; no stretching to ATB resolution. VDA input resize/crop/padding and intrinsics update not fixed.','No depth-confidence/occlusion mask, subject mask or bokeh GT downloaded/generated; all-frame finite positivity is not full annotation validity.','No VDA predictions or calibrated reference in VDA coordinates; not training-ready.',
            'Depth stored in official units; axial-Z versus camera-ray distance convention must be resolved before metric projection/CoC calibration.','Spring sample initial HTTP read timed out; Sintel is starter pipeline data, not final corpus selection.'])
 write_json(ROOT/(config['record_prefix']+'-verification.json'),report)
 # Native RGB plus inverse-depth preview; per-sequence fixed 2-98 range is display only.
 sheet=Image.new('RGB',(5*256,6*250),'white');draw=ImageDraw.Draw(sheet)
 for i,seq in enumerate(config['sequences']):
  lo,hi=seq_inverse_ranges[seq['sequence']]
  for j,t in enumerate([1,7,13,19,25]):
   row=next(x for x in rows if x['sequence']==seq['sequence'] and x['frame']==t)
   with Image.open(ROOT/row['rgb']) as im:
    h=round(im.height*256/im.width);sheet.paste(im.resize((256,h)),(j*256,i*250+20))
   with np.load(ROOT/row['derived_reference']) as a:
    v=a['inverse_depth_m_inv'];v=np.uint8(np.clip((v-lo)/(hi-lo),0,1)*255)
   sheet.paste(Image.fromarray(v).convert('RGB').resize((256,h)),(j*256,i*250+25+h))
   draw.text((j*256+3,i*250+2),f"{seq['sequence']} f{t:04d} {seq['split']}",fill='black')
 sheet.save(derived/'rgb-inverse-depth-contact.png')
 print('PILOT_READY',len(rows),'frames',flush=True)

def audit(config):
 manifest=json.loads((ROOT/(config['record_prefix']+'-manifest.json')).read_text());rows=[]
 for r in manifest['frames']:
  d=depth_read(ROOT/r['depth_m']);mask=d>=1e10
  if mask.any():rows.append(dict(sequence=r['sequence'],frame=r['frame'],pixels=int(mask.sum()),fraction=float(mask.mean()),minimum_flagged_depth=float(d[mask].min()),maximum_flagged_depth=float(d[mask].max())))
 result=dict(date='2026-10-22',diagnostic_threshold_m=1e10,threshold_meaning='Detection of extreme values for review, not an adopted validity or clipping rule',observed_sequences=sorted({x['sequence'] for x in rows}),frames_with_extreme_values=len(rows),rows=rows,consequence='Finite positive numeric validity must NOT be used as final supervision validity. Resolve extreme/far-background values and axial-Z vs ray-distance convention before calibration/training.',raw_and_derived_arrays_unchanged=True)
 write_json(ROOT/(config['record_prefix']+'-depth-convention-audit.json'),result)
 print('DEPTH_CONVENTION_REVIEW',len(rows),'frames',result['observed_sequences'],flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['download','verify','audit']);args=parser.parse_args()
 c=json.loads(CONFIG.read_text());globals()[args.stage](c)
