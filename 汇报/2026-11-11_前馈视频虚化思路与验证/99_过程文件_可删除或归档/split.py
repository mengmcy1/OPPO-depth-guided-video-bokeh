import sys, numpy as np
from PIL import Image
def runs(mask, minlen):
    out=[];start=None
    for i,v in enumerate(mask):
        if v and start is None: start=i
        if not v and start is not None:
            if i-start>=minlen: out.append((start,i))
            start=None
    if start is not None and len(mask)-start>=minlen: out.append((start,len(mask)))
    return out
def tiles(path, minsize=60):
    im=np.asarray(Image.open(path).convert('RGB')).astype(int)
    nw=(im.min(axis=2)<235)
    H,W=nw.shape
    rows=runs(nw.mean(axis=1)>0.25, minsize)
    res=[]
    for r0,r1 in rows:
        band=nw[r0:r1]
        cols=runs(band.mean(axis=0)>0.6, minsize)
        res.append([(r0,r1,c0,c1) for c0,c1 in cols])
    return res
if __name__=='__main__':
    for p in sys.argv[1:]:
        t=tiles(p); print(p, Image.open(p).size)
        for i,row in enumerate(t): print('  row',i,row)
