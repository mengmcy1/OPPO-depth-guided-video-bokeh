"""NumPy surface-footprint and GT-consistent observation sampling helpers."""
import numpy as np

def local_footprint(local,obj,depth,mesh,frame,K,pose):
    out=np.zeros(depth.shape,np.float32)
    for axis in [0,1]:
        delta=np.linalg.norm(np.diff(local,axis=axis),axis=-1);same=np.diff(obj,axis=axis)==0
        delta=np.where(same&np.isfinite(delta),delta,0.);pad=[(0,0),(0,0)];pad[axis]=(0,1);out=np.maximum(out,np.pad(delta,pad))
    for oi,oid in enumerate(mesh['object_ids']):
        at=obj==oid;inv=np.linalg.inv(mesh['object_world_matrices'][frame-1,oi,:3,:3]);factor=np.linalg.norm(inv@pose[:3,:2],axis=0).max()
        out[at]=np.maximum(out[at],depth[at]/K[0,0]*factor)
    return np.maximum(out,1e-6).astype(np.float32)

def sample_identity_rgb(uv,local,ids,gt,rgb,foot):
    """GT-consistent bilinear source colors; no other object's pixel admitted."""
    x=uv[:,0]-.5;y=uv[:,1]-.5;x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);fx=x-x0;fy=y-y0
    total=np.zeros(len(x),np.float64);color=np.zeros((len(x),3),np.float64)
    for dx,dy,ww in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        xx=np.clip(x0+dx,0,511);yy=np.clip(y0+dy,0,511);match=(gt['object_id'][yy,xx]==ids)&(np.linalg.norm(gt['local_surface_point'][yy,xx]-local,axis=-1)<=3*foot[yy,xx]);weight=ww*match;total+=weight;color+=rgb[yy,xx]*weight[:,None]
    good=total>1e-6;color/=np.maximum(total[:,None],1e-30)
    return color.astype(np.float32),good

