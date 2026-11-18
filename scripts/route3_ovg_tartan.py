"""Fetch only the front RGB/depth members of fixed TartanAir V2 windows."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import zipfile

import cv2
import numpy as np
import requests
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/route3-ovg-1a-2026-11-18.json').read_text())
OUT = ROOT/CONFIG['output']/'G'
REPO = 'theairlabcmu/tartanair2'
REVISION = '0d2d145e973832742a2aaa04b7d2ebffc8d82817'
HOST = 'https://huggingface.co/datasets/' + REPO
CV_TO_NED = np.array([[0,0,1],[1,0,0],[0,1,0]], np.float32)


class RangeArchive(io.RawIOBase):
    """Read zip metadata and the chosen members without copying the archive."""
    def __init__(self, url, size):
        self.url, self.size = url, size
        self.position, self.bytes_received = 0, 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET:0, io.SEEK_CUR:self.position, io.SEEK_END:self.size}[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError('Negative zip position')
        return self.position

    def read(self, count=-1):
        start = self.position
        end = min(self.size, start+count) if count>=0 else self.size
        if end<=start:
            return b''
        for attempt in range(4):
            try:
                # The query makes separate range requests distinguishable in
                # HTTP caches. Reject full-object fallback before reading.
                with self.session.get(self.url+f'?download=true&ovg={start}-{end-1}',
                                      headers={'Range':f'bytes={start}-{end-1}'},
                                      timeout=(15,90),stream=True) as r:
                    expected = f'bytes {start}-{end-1}/{self.size}'
                    if r.status_code!=206 or r.headers.get('Content-Range')!=expected:
                        raise OSError(f'HTTP {r.status_code} or incorrect Content-Range for {self.url}')
                    data = r.content
                if len(data)!=end-start:
                    raise OSError('Incomplete ZIP member range')
                self.bytes_received+=len(data)
                self.position=end
                return data
            except (requests.RequestException,OSError):
                if attempt==3:
                    raise
                time.sleep(1.5*(attempt+1))

    def close(self):
        self.session.close()
        super().close()


def json_write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.part')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def depth_from_png(path):
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim!=3 or raw.shape[-1]!=4 or raw.dtype!=np.uint8:
        raise ValueError(f'Invalid packed depth at {path}')
    return np.ascontiguousarray(raw).view('<f4')[...,0]


def members(zf, kind):
    result={}
    for name in zf.namelist():
        p=PurePosixPath(name)
        trajectory=next((x for x in p.parts if re.fullmatch(r'P\d+',x)),None)
        stem=re.match(r'^(\d+)_',p.name)
        if trajectory and stem and p.suffix=='.png' and p.parent.name==kind:
            result.setdefault(trajectory,{})[int(stem[1])]=name
    return result


def windows(rgb,depth,poses,n,num):
    possible=[]
    for run,files in sorted(rgb.items()):
        if run not in depth or run not in poses:
            continue
        shared=set(files)&set(depth[run])
        for f in sorted(shared):
            if all(f+j in shared for j in range(n)):
                possible.append((run,f))
    if len(possible)<num:
        raise ValueError('No aligned contiguous RGB/depth/pose windows')
    chosen=[]
    for pos in np.linspace(.2,.8,num):
        for run,start in sorted(possible,key=lambda x:abs(possible.index(x)/(len(possible)-1)-pos)) if len(possible)<100 else [possible[int(pos*(len(possible)-1))]]:
            if all(run!=r or abs(start-s)>=n for r,s in chosen):
                chosen.append((run,start));break
    if len(chosen)!=num:
        raise ValueError('Not enough disjoint windows')
    return chosen


def manifest_window(sheet,env,run,start,ids,images,depths,poses):
    dest=OUT/'windows'/f'{env}-{run}-{start:06d}'
    dest.mkdir(parents=True,exist_ok=True)
    items=[]
    for i,frame in enumerate(ids):
        rgb_dest=dest/f'frame-{i:02d}.png'
        depth_dest=dest/f'depth-{i:02d}.png'
        for archive,name,p in [(images[0],images[1][frame],rgb_dest),
                               (depths[0],depths[1][frame],depth_dest)]:
            if not p.exists():
                raw=archive.read(name)  # zipfile checks each stored CRC
                with p.open('xb') as f:
                    f.write(raw)
        z=depth_from_png(depth_dest)
        if z.shape!=(640,640) or np.isfinite(z).mean()<.95 or np.median(z)<=0:
            raise ValueError(f'Unexpected TartanAir depth payload {depth_dest}')
        r=poses[frame]
        pose=np.eye(4,dtype=np.float32)
        pose[:3,:3]=Rotation.from_quat(r[3:]).as_matrix()@CV_TO_NED
        pose[:3,3]=r[:3]
        items.append(dict(frame=frame,image=str(rgb_dest.relative_to(ROOT)),depth=str(depth_dest.relative_to(ROOT)),
                          c2w=pose.tolist(),K=[[320,0,320],[0,320,320],[0,0,1]]))
    data=dict(tier='G',scene=dest.name,environment=env,trajectory=run,rows=items,
              depth_format='little-endian float32 stored in 4 PNG bytes, official OpenCV order',
              pose_format='NED xyz and xyzw converted to OpenCV camera-to-world',**sheet)
    json_write(dest/'manifest.json',data)
    return data


def fetch(env):
    start_time=time.monotonic()
    raw=OUT/'acquisition'/f'{env}.json'
    if raw.exists():
        return json.loads(raw.read_text())
    cfg=CONFIG['G']
    folder=f"{env}/{cfg['difficulty']}"
    api='https://huggingface.co/api/datasets/'+REPO+f'/tree/{REVISION}/{folder}'
    r=requests.get(api,timeout=30);r.raise_for_status()
    listing={v['path']:v['size'] for v in r.json()}
    root=HOST+'/resolve/'+REVISION+'/'
    names=[folder+'/'+x+'_lcam_front.zip' for x in ['image','depth']]
    if shutil.disk_usage(OUT).free/2**30 < 60:
        raise RuntimeError('Data disk below 60 GiB')
    with RangeArchive(root+names[0],listing[names[0]]) as ri, RangeArchive(root+names[1],listing[names[1]]) as rd:
        with zipfile.ZipFile(ri) as image,zipfile.ZipFile(rd) as depth:
            ims=members(image,'image_lcam_front')
            deps=members(depth,'depth_lcam_front')
            pose_members={}
            for z in (image,depth):
                for path in z.namelist():
                    p=PurePosixPath(path)
                    if p.name=='pose_lcam_front.txt':
                        run=next((x for x in p.parts if re.fullmatch(r'P\d+',x)),None)
                        if run:
                            pose_members.setdefault(run,(z,path))
            print('TARTAN_ZIP',env,len(ims),len(deps),len(pose_members),image.namelist()[:3],flush=True)
            selections=windows(ims,deps,pose_members,cfg['frames_per_window'],cfg['windows_per_environment'])
            result=[]
            for run,first in selections:
                zf,name=pose_members[run]
                poses=np.loadtxt(io.BytesIO(zf.read(name)))
                fids=list(range(first,first+cfg['frames_per_window']))
                result.append(manifest_window(dict(dataset=REPO,revision=REVISION,pose_member=name,source_archives=names,
                                                   license='CC-BY-4.0 per TartanAir V2 official documentation',
                                                   license_url='https://tartanair.org/'),env,run,first,fids,
                                              (image,ims[run]),(depth,deps[run]),poses))
            record=dict(environment=env,windows=[x['scene'] for x in result],frames=sum(len(x['rows']) for x in result),
                        HTTP_range_bytes=ri.bytes_received+rd.bytes_received,seconds=time.monotonic()-start_time,
                        archive_bytes=sum(listing[k] for k in names),revision=REVISION)
            json_write(raw,record)
            return record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as group:
        done=list(group.map(fetch,CONFIG['G']['environments']))
    json_write(OUT/'acquisition.json',dict(sources=done,dataset=REPO,revision=REVISION,
               download_scope='front RGB, depth and pose for contiguous windows; byte ranges for zip metadata',
               license_url='https://tartanair.org/',hf_card_metadata_license='bsd-3-clause',
               dataset_terms='CC BY 4.0 (official project documentation); provide attribution for data use'))


if __name__=='__main__':
    main()
