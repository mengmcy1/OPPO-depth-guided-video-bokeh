"""Range-only acquisition of uniformly sampled TartanAir V2 windows."""
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import struct
from threading import Lock, local
import time
import zipfile
import zlib

import numpy as np
import requests

from route3_ovg_r2_common import CONFIG, OUT, guard
import route3_ovg_tartan as official
from route3_ovg_subpixel import save_json

MAX_ENVS=4
PER_ENV_DOWNLOADS=8


class DirectZip(official.RangeArchive):
    """Use one signed CDN redirect in memory; validate all byte intervals and CRCs."""
    def __init__(self, url, size):
        super().__init__(url, size)
        self.thread=local()
        self.lock=Lock()
        self.direct=None
        self.refresh()

    def refresh(self):
        with self.lock:
            r=self.session.get(self.url+'?download=true&ovg-r2=resource',
                               allow_redirects=False,timeout=25,stream=True)
            try:
                self.direct=r.headers.get('location') if r.is_redirect else None
                if not self.direct:
                    raise OSError('Public ZIP resource did not return the expected CDN redirect')
            finally:
                r.close()

    def read_at(self,start,end):
        if not hasattr(self.thread,'session'):
            self.thread.session=requests.Session()
        session=self.thread.session
        for attempt in range(5):
            try:
                with session.get(self.direct,headers={'Range':f'bytes={start}-{end-1}'},
                                 timeout=(12,80),stream=True) as response:
                    status=response.status_code
                    if status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end-1}/{self.size}':
                        raise OSError(f'Unexpected range response status {status}')
                    data=response.content
                if len(data)!=end-start:
                    raise OSError('Incomplete byte interval')
                with self.lock:
                    self.bytes_received+=len(data)
                return data
            except (OSError,requests.RequestException):
                if attempt==4:
                    raise OSError(f'Public ZIP range failed {start}-{end-1} of {self.url}') from None
                if attempt==2:
                    self.refresh()
                time.sleep(.9*(attempt+1))

    def read(self,count=-1):
        start=self.position
        end=min(self.size,start+count) if count>=0 else self.size
        if start>=end:
            return b''
        data=self.read_at(start,end)
        self.position=end
        return data

    def read_member(self, member):
        start=member.header_offset
        guess=min(self.size,start+30+len(member.filename.encode())+member.compress_size+128)
        data=self.read_at(start,guess)
        if data[:4]!=b'PK\x03\x04':
            raise OSError('Invalid ZIP local header')
        n,x=struct.unpack('<HH',data[26:30])
        offset=30+n+x
        if offset+member.compress_size>len(data):
            data+=self.read_at(guess,start+offset+member.compress_size)
        payload=data[offset:offset+member.compress_size]
        if member.compress_type==zipfile.ZIP_DEFLATED:
            payload=zlib.decompress(payload,-15)
        elif member.compress_type!=zipfile.ZIP_STORED:
            raise ValueError(f'Unsupported public ZIP method {member.compress_type}')
        if len(payload)!=member.file_size or zlib.crc32(payload)!=member.CRC:
            raise OSError(f'ZIP member CRC/size mismatch {member.filename}')
        return payload


class Members:
    def __init__(self,remote,index):
        self.reader,self.index=remote,index

    def read(self,path):
        return self.reader.read_member(self.index.getinfo(path))


def select(rgb,depth,pose_keys,n,amount,old):
    candidates=[]
    for run,rows in sorted(rgb.items()):
        if run not in pose_keys or run not in depth:
            continue
        shared=set(rows)&set(depth[run])
        for f in sorted(shared):
            if f%n==0 and all(f+i in shared for i in range(n)):
                if all(run!=other or abs(f-start)>=n for other,start in old):
                    candidates.append((run,f))
    if len(candidates)<amount:
        raise ValueError(f'Only {len(candidates)} valid windows for requested {amount}')
    indexes=np.linspace(0,len(candidates)-1,amount+2)[1:-1].astype(int)
    return [candidates[i] for i in indexes]


def fetch(env):
    guard()
    cached=OUT/'G/acquisition'/f'{env}.json'
    if cached.exists() and json.loads(cached.read_text()).get('complete'):
        return json.loads(cached.read_text())
    start=time.monotonic()
    cfg=CONFIG['G_round2']
    folder=f'{env}/Data_easy'
    href=f'https://huggingface.co/api/datasets/{official.REPO}/tree/{official.REVISION}/{folder}'
    r=requests.get(href,timeout=30)
    if r.status_code==404:
        return dict(environment=env,frames=0,complete=False,reason='Pinned upstream directory HTTP 404')
    r.raise_for_status()
    sizes={x['path']:x['size'] for x in r.json()}
    archives=[f'{folder}/{name}_lcam_front.zip' for name in ('image','depth')]
    prefix=f'{official.HOST}/resolve/{official.REVISION}/'
    # Keep the original six ready shards linked into this run's inventory.
    original=[]
    for p in (OUT/'G/windows').glob(f'{env}-*/manifest.json'):
        if not p.parent.is_symlink():
            continue
        row=json.loads(p.read_text())
        original.append((row['trajectory'],row['rows'][0]['frame']))
    targets=[]
    print('G_ENV_START',env,flush=True)
    with DirectZip(prefix+archives[0],sizes[archives[0]]) as ri, DirectZip(prefix+archives[1],sizes[archives[1]]) as rd:
        with zipfile.ZipFile(ri) as zi,zipfile.ZipFile(rd) as zd:
            ims=official.members(zi,'image_lcam_front')
            deps=official.members(zd,'depth_lcam_front')
            pose_files={}
            for archive,zf in ((ri,zi),(rd,zd)):
                for member in zf.infolist():
                    if not member.filename.endswith('/pose_lcam_front.txt'):
                        continue
                    run=next((x for x in member.filename.split('/') if re.fullmatch(r'P\d+',x)),None)
                    if run:
                        pose_files.setdefault(run,(archive,member))
            quota=cfg['windows_per_environment']-len(original)
            n=cfg['frames_per_window']
            windows=select(ims,deps,pose_files,n,quota,original)
            poses={}
            for run in sorted({run for run,_ in windows}):
                a,p=pose_files[run]
                poses[run]=np.loadtxt(io.BytesIO(a.read_member(p)))

            def one(item):
                run,f=item
                name=f'{env}-{run}-{f:06d}'
                stable=OUT/'G/windows'/name/'manifest.json'
                if not stable.exists():
                    guard()
                    official.manifest_window(
                        dict(dataset=official.REPO,revision=official.REVISION,
                             pose_member=pose_files[run][1].filename,source_archives=archives,
                             license='CC BY 4.0 (author documentation)',license_url='https://tartanair.org/'),
                        env,run,f,list(range(f,f+n)),(Members(ri,zi),ims[run]),
                        (Members(rd,zd),deps[run]),poses[run])
                    print('G_SHARD_READY',env,name,flush=True)
                return name

            with ThreadPoolExecutor(max_workers=PER_ENV_DOWNLOADS) as workers:
                for f in as_completed([workers.submit(one,s) for s in windows]):
                    targets.append(f.result())
        result=dict(environment=env,old_windows=len(original),new_windows=sorted(targets),
                    frames=(len(original)+len(targets))*n,complete=True,seconds=time.monotonic()-start,
                    range_bytes=ri.bytes_received+rd.bytes_received,archive_bytes=sum(sizes[x] for x in archives),
                    revision=official.REVISION)
        save_json(cached,result)
        print('G_ENV_READY',env,result['frames'],round(result['seconds'],1),flush=True)
        return result


def main():
    guard()
    official.OUT=OUT/'G'
    results=[]
    with ThreadPoolExecutor(max_workers=MAX_ENVS) as workers:
        futures={workers.submit(fetch,e):e for e in CONFIG['G_round2']['environments']}
        for f in as_completed(futures):
            env=futures[f]
            try:
                results.append(f.result())
            except Exception as error:
                # Keep every already decoded, CRC-checked complete window.
                # The affected environment alone is eligible for a retry.
                results.append(dict(environment=env,complete=False,reason=str(error),frames=0))
                print('G_ENV_ERROR',env,type(error).__name__,str(error),flush=True)
            save_json(OUT/'G/acquisition.json',dict(environments=results,dataset=official.REPO,
                      revision=official.REVISION,official_terms='CC BY 4.0',
                      href='https://tartanair.org/',direct_selected_ranges=True,
                      downloads='only ZIP directory, poses and contiguous front RGB/depth windows'))
    print('G_FETCH_DONE',len(results),flush=True)


if __name__=='__main__':
    main()
