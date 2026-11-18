"""Analytic textured plane sequence and a mesh/analytic agreement check."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

import cv2
import numpy as np

from route3_ovg_subpixel import CONFIG, OUT, ROOT, offsets, save_json, save_npz, two_layers

REC=ROOT/'03_实验记录/records'
TEST=REC/'route3-ovg-1a-2026-11-18-independent-test.json'
PH='https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k'


def test_manifest():
    if TEST.exists():
        return json.loads(TEST.read_text())
    names=['weathered_planks','red_brick_03','concrete_wall_006','sandstone_blocks_05',
           'wood_planks_grey','wood_floor_deck']
    categories=['broad','multilayer','thin-rod','perforated','grazing','weak-texture']
    cases=[]
    for j,category in enumerate(categories):
        for i in range(3):
            seed=774618+j*100+i
            rng=np.random.default_rng(seed)
            paths=[names[(j+i)%6],names[(j+i+2)%6]]
            phases=rng.uniform(-.65,.65,2)
            base=float(rng.uniform(.16,.46))
            center=np.array([0,0,0.])
            t=np.linspace(0,1,13)
            translation=np.stack([base*(t-.5), .04*np.sin(2*np.pi*t+phases[0]),
                                  .06*np.sin(np.pi*t+phases[1])],-1)
            aim=np.array([.05*(i-1),0.,2.8+j*.19])
            cases.append(dict(id=f'ovg-test-{category}-{i+1}',category=category,seed=seed,
                              render_state='sealed, unrendered',
                              asset_sources=[f'https://polyhaven.com/a/{x}' for x in paths],
                              geometry_recipe=dict(provider='project procedural test-only asset family',
                                    asset_id=f'ovg-test-mesh-{category}-{seed}',
                                    foreground_m=round(1.25+.1*i,3),rear_surface_m=round(4.4+.5*j+.23*i,3),
                                    category=category,asset_combination_index=i),
                              camera=dict(translation_m=(center+translation).tolist(),
                                          look_at_m=aim.tolist(),world_up=[0,-1,0],
                                          resolution=[512,512],focal_mm=50,sensor_mm=36)))
    card=dict(id='ovg-test-card-1',seed=775991,asset_ids=['painted_plaster_wall','sandstone_blocks_05'],
              category='photographic planar occlusion and never-seen negative',frames=13,
              camera=dict(translation_formula='x=.34*sin(pi*(t-.5)), y=.025*sin(2*pi*t), z=.03*sin(pi*t), t=i/12',
                          opencv_rotation=np.eye(3).tolist(),focal_mm=50,sensor_mm=36),
              asset_sources=['https://polyhaven.com/a/painted_plaster_wall','https://polyhaven.com/a/sandstone_blocks_05'])
    result=dict(sealed_before_train=True,protocol_revision=CONFIG['protocol_revision'],asset_license='CC0',
                three_dimensional=cases,card=[card],
                forbidden_train=dict(seeds=[x['seed'] for x in cases]+[card['seed']],
                    asset_ids=sorted(set(names+card['asset_ids'])),
                    scene_ids=[x['id'] for x in cases]+[card['id']]),
                scope='independent scenes and source assets; no rendering or model selection')
    save_json(TEST,result)
    return result


def load_texture(asset, dest):
    import requests
    file=dest/(asset+'.jpg')
    url=f'{PH}/{asset}/{asset}_diff_1k.jpg'
    if not file.exists():
        r=requests.get(url+'?route3_ovg_card=1',timeout=30)
        r.raise_for_status()
        if len(r.content)<10_000 or cv2.imdecode(np.frombuffer(r.content,np.uint8),cv2.IMREAD_COLOR) is None:
            raise ValueError(f'Invalid photo download {asset}')
        file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(r.content)
    a=cv2.imread(str(file))[...,::-1].astype(np.float32)/255
    a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
    return a,url


def cast(rays, origin, cards):
    t=(cards[:,0]-origin[2])/np.maximum(rays[...,2,None],1e-8)
    x=origin[0]+rays[...,0,None]*t
    y=origin[1]+rays[...,1,None]*t
    visible=(t>0)&(x>=cards[:,1])&(x<cards[:,2])&(y>=cards[:,3])&(y<cards[:,4])
    slot=visible.argmax(-1)
    world=np.stack([x,y,np.broadcast_to(cards[:,0],x.shape)],-1)
    point=np.take_along_axis(world,slot[...,None,None],axis=-2)[...,0,:]
    valid=visible.any(-1)
    return dict(depth=np.where(valid,point[...,2]-origin[2],0).astype(np.float32),
                world=point.astype(np.float32),object=np.where(valid,slot+1,0).astype(np.uint8),
                visible=visible,world_all=world)


def lens_offsets():
    i=np.arange(8); theta=i*np.pi*(3-np.sqrt(5)); r=np.sqrt((i+.5)/8)
    a=np.stack([r*np.cos(theta),r*np.sin(theta)],-1)
    return np.c_[np.concatenate([a,-a],0)*.05/(2*1.4),np.zeros(16)]


def uv(coords,origin,K):
    q=(coords-origin)@K.T
    return q[...,:2]/np.maximum(q[...,2:],1e-8)


def visibility(points,objects,origin,K,cards,size):
    xy=uv(points,origin,K)
    inside=((xy>=0)&(xy<size)).all(-1)
    hit=cast(points-origin,origin,cards)
    valid=inside&(points[...,2]>origin[2])&(objects>0)
    visible=valid&(hit['object']==objects)&(abs(hit['world'][...,2]-points[...,2])<1e-5)
    return visible,valid,hit,xy


def mesh(cards, K, poses, path):
    values=dict(object_names=np.array([f'plane_{i}' for i in range(3)]),object_ids=np.arange(1,4),
                foreground_object_ids=np.array([1,2]),dynamic_object_ids=np.array([],np.int16),
                intrinsics=K* np.array([[2.,2.,2.],[2.,2.,2.],[1.,1.,1.]]),
                camera_c2w_opencv=poses,object_world_matrices=np.tile(np.eye(4),(len(poses),3,1,1)),
                triangle_object_id=np.repeat(np.arange(1,4),2),
                triangle_polygon_id=np.tile(np.arange(2),3), input_frames=np.arange(1,len(poses)+1))
    for i,(z,x0,x1,y0,y1) in enumerate(cards,1):
        values[f'object_{i}_vertices']=np.array([[x0,y0,z],[x0,y1,z],[x1,y1,z],[x1,y0,z]])
        values[f'object_{i}_faces']=np.array([[0,1,2],[0,2,3]])
    save_npz(path,**values)


def generate():
    split=json.loads(TEST.read_text()) if TEST.exists() else test_manifest()
    spec=CONFIG['C']
    forbidden=split['forbidden_train']
    assets=spec['training_assets']
    if set(assets)&set(forbidden['asset_ids']) or spec['seed'] in forbidden['seeds']:
        raise ValueError('Card asset or scene overlaps sealed test inventory')
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as workers:
        loaded=list(workers.map(lambda name:load_texture(name,OUT/'C/assets'),assets))
    size=spec['size']
    rng=np.random.default_rng(spec['seed'])
    K=np.array([[size*50/36,0,size/2],[0,size*50/36,size/2],[0,0,1]],np.float64)
    grid=np.arange(size)+.5
    pix=np.stack([*np.meshgrid(grid,grid),np.ones((size,size))],-1)
    ray=pix@np.linalg.inv(K).T
    rays=(pix[...,None,:]+offsets(4))@np.linalg.inv(K).T
    # Two foreground occluders, a broad background and noninteger image edges.
    jitter=float(rng.uniform(-.012,.012))
    cards=np.array([[1.4,-.25+jitter,.055+jitter,-.29,.25],[2.7,.03,.57,-.56,.59],[5.,-10.,10.,-10.,10.]])
    t=np.linspace(-1,1,spec['frames'])
    origins=np.stack([.24*t,.046*np.sin(np.pi*t),.035*np.sin(1.7*t)],-1)
    poses=np.tile(np.eye(4),(len(t),1,1))
    poses[:,:3,3]=origins
    dest=OUT/'C/windows'/f'card-{spec["seed"]}'
    dest.mkdir(parents=True,exist_ok=True)
    mesh(cards,K,poses,dest/'scene-geometry.npz')
    rows, stats=[],[]
    for f,(position,pose) in enumerate(zip(origins,poses)):
        h=cast(rays,position,cards)
        obj=h['object']; normals=np.zeros((*obj.shape,3),np.float32);normals[...,2]=np.where(obj>0,-1.,0.)
        label=two_layers(h['depth'],normals,obj)
        p=dest/f'labels-{f:02d}.npz'
        save_npz(p,**label)
        rgb=np.zeros((*obj.shape,3),np.float32)
        for i,((tex,_),card) in enumerate(zip(loaded,cards),1):
            drawn=obj==i
            # Tile at 1.25 metres, including the rear photo outside the view.
            coord=np.mod(h['world'][drawn,:2]/1.25,1.)
            xy=(coord*np.array([tex.shape[1]-1,tex.shape[0]-1])).astype(np.float32)
            if len(xy):
                out=[]
                for part in np.array_split(xy,int(np.ceil(len(xy)/16000))):
                    out.append(cv2.remap(tex,part[:,None,0],part[:,None,1],cv2.INTER_LINEAR)[:,0])
                rgb[drawn]=np.concatenate(out)
        render=rgb.mean(-2)
        display=np.where(render<=.0031308,12.92*render,1.055*render**(1/2.4)-.055)
        image=dest/f'frame-{f:02d}.png'
        if not image.exists():
            cv2.imwrite(str(image),np.rint(display.clip(0,1)[...,::-1]*255).astype(np.uint8))
        central=cast(ray,position,cards)
        peeled=central['visible'].copy()
        np.put_along_axis(peeled,(central['object'].astype(np.int64)-1)[...,None],False,axis=-1)
        first_hidden=peeled.argmax(-1)
        hp=np.take_along_axis(central['world_all'],first_hidden[...,None,None],axis=-2)[...,0,:]
        good=peeled.any(-1)
        save_npz(dest/'peeled'/f'f{f:02d}.npz',hidden_valid=good,world=hp.astype(np.float32),
                 depth=np.where(good,hp[...,2]-position[2],0).astype(np.float32),
                 surface_id=np.where(good,first_hidden+1,0).astype(np.uint8))
        # Stable surface identity is plane ID plus continuous in-plane xy.
        for focus in (1.4,5.):
            ray_labels=[]
            for lens in lens_offsets():
                light=cast(ray*focus-lens,position+lens,cards)
                points,objects=light['world'],light['object']
                _,valid,first,xy=visibility(points,objects,position,K,cards,size)
                hidden=valid&(first['world'][...,2]+1e-5<points[...,2])
                bit=np.zeros(hidden.shape,np.uint16)
                for j,other in enumerate(origins):
                    if j==f:
                        continue
                    visible,*_=visibility(points,objects,other,K,cards,size)
                    bit|=(visible&hidden).astype(np.uint16)*(1<<j)
                ray_labels.append(dict(surface_id=objects,world=points,axial_depth=light['depth'],
                          center_projection=xy.astype(np.float32),valid=valid,hidden=hidden,
                          visible_source_frames=bit,recoverable=bit>0,never_seen=hidden&(bit==0)))
            save_npz(dest/'K16-aperture'/f'f{f:02d}-N1.4-F{focus:g}.npz',
                     **{k:np.stack([d[k] for d in ray_labels]) for k in ray_labels[0]})
            stats.append(dict(frame=f,focus_m=focus,hidden_hits=sum(int(r['hidden'].sum()) for r in ray_labels),
                          recoverable_hits=sum(int(r['recoverable'].sum()) for r in ray_labels),
                          never_seen_hits=sum(int(r['never_seen'].sum()) for r in ray_labels)))
        rows.append(dict(frame=f,image=str(image.relative_to(ROOT)),label=str(p.relative_to(ROOT)),
                         c2w=pose.tolist(),K=K.tolist()))
    save_json(dest/'manifest.json',dict(tier='C',scene=dest.name,rows=rows,seed=spec['seed'],
                         assets=[dict(asset=x,url=y[1],license='CC0') for x,y in zip(assets,loaded)],
                         static_scene=True,completed=True,seconds=time.monotonic()-start,
                         centers=origins.tolist(),cards=cards.tolist(),lens=stats,
                         independent_optical_reference='1b', K16_separate=True,
                         motion='camera translation; world-stable front and back planes'))


def check():
    from route3_ovg_subpixel import intersect
    from route3_blender_b2_geometry import World
    spec=CONFIG['C']
    path=OUT/'C/windows'/f'card-{spec["seed"]}'
    meta=json.loads((path/'manifest.json').read_text())
    frame=6
    world=World(path/'scene-geometry.npz',spec['size'])
    origin=np.array(meta['centers'][frame])
    cards=np.array(meta['cards'])
    yy,xx=np.mgrid[:world.size,:world.size]
    xy=np.c_[xx.ravel(),yy.ravel()]
    pick=np.random.default_rng(spec['seed']).choice(len(xy),8192,replace=False)
    d,n,objects=intersect(world,frame+1,xy[pick],4)
    analytic=cast((np.c_[xy[pick]+.5,np.ones(len(pick))][:,None]+offsets(4))@np.linalg.inv(world.K).T,
                  origin,cards)
    lens=lens_offsets()[3]
    ray=np.c_[xy[pick]+.5,np.ones(len(pick))]@np.linalg.inv(world.K).T
    hit,faces=world.cast(frame+1,origin+lens,ray*1.4-lens)
    analytic_lens=cast(ray*1.4-lens,origin+lens,cards)
    ids=np.where(faces>=0,world.tri_obj[np.maximum(faces,0)],0)
    delta=abs(d-analytic['depth'])
    value=dict(pinhole_checked_rays=int(d.size),lens_checked_rays=len(pick),
               pinhole_identity_agreement=float((objects==analytic['object']).mean()),
               lens_identity_agreement=float((ids==analytic_lens['object']).mean()),
               max_depth_error_same_surface_m=float(delta[objects==analytic['object']].max()),
               deterministic_seed=spec['seed'],frames_compared=[frame], fstop=1.4,
               method='same geometry through analytic planes and B mesh/Embree pinhole and off-axis ray conventions',
               acceptance=dict(min_identity=.999,max_depth_m=1e-4))
    value['passed']=value['pinhole_identity_agreement']>=.999 and value['lens_identity_agreement']>=.999 and value['max_depth_error_same_surface_m']<=1e-4
    save_json(OUT/'C/mesh-check.json',value)
    if not value['passed']:
        raise ValueError('Card analytic and Blender mesh conventions disagree')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['seal','generate','check'])
    {'seal':test_manifest,'generate':generate,'check':check}[p.parse_args().stage]()
