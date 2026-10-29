"""Resume official public Stage0 weights with validated HTTP range chunks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]


def fetch(url,path,size,expected=None):
    if path.exists():
        assert path.stat().st_size==size
        print('REUSE',path.name,flush=True);return
    part=path.with_suffix(path.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
    assert offset<=size
    chunks=path.parent/(path.name+'-chunks');chunks.mkdir(exist_ok=True)
    ranges=[(a,min(a+32*1024**2,size)-1) for a in range(offset,size,32*1024**2)]
    def one(pair):
        a,b=pair;p=chunks/f'{a}-{b}.bin'
        if p.exists():assert p.stat().st_size==b-a+1;return p
        for attempt in range(6):
            try:
                r=requests.get(url+f'?stage0_range={a}-{b}&attempt={attempt}',headers={'Range':f'bytes={a}-{b}'},timeout=(30,90))
                r.raise_for_status();assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{size}',(r.status_code,r.headers.get('Content-Range'))
                assert len(r.content)==b-a+1
                temp=p.with_suffix('.part');temp.write_bytes(r.content);temp.rename(p)
                print('CHUNK',path.name,a,b,flush=True);return p
            except (requests.RequestException,AssertionError) as exc:
                print('RETRY',path.name,a,attempt,type(exc).__name__,flush=True)
                if attempt==5:raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=4) as pool:pieces=list(pool.map(one,ranges))
    with part.open('ab') as f:
        for p in pieces:f.write(p.read_bytes())
    assert part.stat().st_size==size
    h=hashlib.sha256()
    with part.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    digest=h.hexdigest();assert expected is None or expected==digest
    part.rename(path);print('COMPLETE',path.name,size,digest,flush=True)
    (path.parent/'download.json').write_text(json.dumps(dict(url=url,bytes=size,sha256=digest,official_expected_sha256=expected,resumed_prefix_bytes=offset),indent=2)+'\n')


if __name__=='__main__':
    c=json.loads((ROOT/'configs/v2-stage0-20261029.json').read_text())
    v=c['models']['VGGT'];fetch('https://huggingface.co/facebook/VGGT-1B/resolve/'+v['HF_revision']+'/model.pt',ROOT/v['weight'],5026874952,v['sha256'])
    p=c['models']['P0'];fetch(p['url'],ROOT/p['weight'],1217607321)
