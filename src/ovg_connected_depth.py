"""Tear-aware inverse-depth subdivision on the original pixel grid."""
import numpy as np


DIRECTIONS = ((0,1),(1,0),(1,1),(1,-1))


def edge_slices(dy,dx):
    return (slice(max(0,-dy),-dy if dy>0 else None),
            slice(max(0,-dx),-dx if dx>0 else None)), (
                slice(max(0,dy),dy if dy<0 else None),
                slice(max(0,dx),dx if dx<0 else None))

def relative_jump(a,b):
    maximum = np.maximum(a,b)
    return np.divide(np.abs(a-b),maximum,out=np.zeros_like(a),where=maximum>0)


def rule_tears(inverse,tau):
    labels = np.full((4,*inverse.shape),-2,np.int8)
    for k,(dy,dx) in enumerate(DIRECTIONS):
        a,b = edge_slices(dy,dx)
        labels[k][a] = relative_jump(inverse[a],inverse[b])>tau
    return labels


def neighbor(array,dy,dx,fill=0):
    a,b = edge_slices(dy,dx)
    shifted = np.full_like(array,fill)
    shifted[a] = array[b]
    return shifted


def connection(links,dy,dx):
    if (dy,dx) in DIRECTIONS:
        return links[DIRECTIONS.index((dy,dx))]
    return neighbor(links[DIRECTIONS.index((-dy,-dx))],dy,dx,False)


def subdivide(inverse,tears):
    """Return H×W×5×5 inverse depths and per-sample interpolation mode.

    Incomplete triangles use a legal edge incident to the parent pixel. The
    edge with the largest positive orthogonal projection is selected. This
    falls back to the parent where the offset has no projection on any
    reachable edge, without extrapolating across an image or depth boundary.
    Index 0 is constant, 1 is linear, 2 is triangular, 3 is the fixed center.
    """
    if inverse.ndim!=2 or tears.shape!=(4,*inverse.shape) or not np.all(np.isfinite(inverse)&(inverse>0)):
        raise ValueError("Subpixel interpolation needs finite positive depth and four native edge maps")
    links = tears==0
    valid = np.ones_like(inverse,bool)
    values = np.empty((*inverse.shape,5,5),np.float32)
    modes = np.empty(values.shape,np.uint8)
    directions = (*DIRECTIONS,*[(-y,-x) for y,x in DIRECTIONS])
    legal = {d:connection(links,*d) for d in directions}
    adjacent = {d:neighbor(inverse,*d) for d in directions}
    points = (-.4,-.2,0.,.2,.4)
    for sy in (-1,1):
        for sx in (-1,1):
            q,r,s = (adjacent[d] for d in ((0,sx),(sy,0),(sy,sx)))
            h,v,diag = (legal[d] for d in ((0,sx),(sy,0),(sy,sx)))
            block = neighbor(valid,sy,sx,False)
            own_gap,other_gap = abs(inverse-s),abs(q-r)
            through_owner = (own_gap<other_gap)|((own_gap==other_gap)&(sy==sx))
            anti = neighbor(legal[sy,-sx],0,sx,False)
            lower = block&~through_owner&h&v&anti
            along_h = block&through_owner&diag&h&neighbor(v,0,sx,False)
            along_v = block&through_owner&diag&v&neighbor(h,sy,0,False)
            iy = [i for i,p in enumerate(points) if (1 if p>=0 else -1)==sy]
            ix = [i for i,p in enumerate(points) if (1 if p>=0 else -1)==sx]
            for i in iy:
                for j in ix:
                    dy,dx = points[i],points[j]
                    if i==j==2:
                        values[...,i,j],modes[...,i,j] = inverse,3
                        continue
                    u,w = abs(dx),abs(dy)
                    best = np.zeros_like(inverse)
                    estimate = inverse.copy()
                    for ey,ex in directions:
                        dot = dy*ey+dx*ex
                        if dot<=0:
                            continue
                        length = ex*ex+ey*ey
                        projected = dot/length**.5
                        choose = legal[ey,ex]&(projected>best)
                        estimate[choose] = ((1-dot/length)*inverse[choose]+dot/length*adjacent[ey,ex][choose])
                        best[choose] = projected
                    mode = (best>0).astype(np.uint8)
                    if u>=w:
                        tri,vertex,weight = along_h,q,u-w
                    else:
                        tri,vertex,weight = along_v,r,w-u
                    estimate[tri] = (1-max(u,w))*inverse[tri]+weight*vertex[tri]+min(u,w)*s[tri]
                    estimate[lower] = (1-u-w)*inverse[lower]+u*q[lower]+w*r[lower]
                    mode[tri|lower] = 2
                    values[...,i,j],modes[...,i,j] = estimate,mode
    tl,tr,bl,br = inverse[:-1,:-1],inverse[:-1,1:],inverse[1:,:-1],inverse[1:,1:]
    diagonal = abs(tl-br)<=abs(tr-bl)
    selection = dict(NW_SE=int(diagonal.sum()),NE_SW=int((~diagonal).sum()))
    return values,modes,selection
