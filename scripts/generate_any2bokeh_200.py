"""Public-generator-style inputs for reproduction, with explicit Cycles fallback differences."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synthetic.common import srgb_to_linear,linear_to_srgb

def build_config(path):
    rng=np.random.RandomState(20)
    am=sorted((ROOT/'data/raw/AM-2k/train/fg').glob('*.png'))
    vm=sorted((ROOT/'data/raw/VideoMatte240K_HEVC/VideoMatte240K/train/fgr').glob('*.mp4'))
    bg=sorted((ROOT/'data/raw/BG-20k/train').glob('*.jpg'))
    assets=[{'kind':'AM','rgb':str(p.relative_to(ROOT)),'alpha':str(p.relative_to(ROOT)).replace('/fg/','/mask/'),'source_id':'AM-2k:'+p.stem} for p in am]
    for p in vm:
        cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if n<97:continue
        assets.append({'kind':'VM','rgb':str(p.relative_to(ROOT)),'alpha':str(p.relative_to(ROOT)).replace('/fgr/','/pha/'),'source_id':'VideoMatte240K:train:'+p.stem,'start':int(rng.randint(0,n-96)),'stride':4})
    scenes=[]
    for i in range(200):
        b=bg[int(rng.randint(len(bg)))];objs=[assets[int(j)] for j in rng.randint(len(assets),size=4)]
        scenes.append({'id':f'repro_{i:03d}','seed':20+i,'background_file':str(b.relative_to(ROOT)),'background_source_id':'BG-20k:'+b.stem,'objects':objs,'focus_object_depth_rank':int(rng.randint(4)),'K':int(rng.choice([4,8,16,32]))})
    cfg={'name':'Reproduced Any-to-Bokeh-style 200-video Test Set','status':'FIXED_BEFORE_INFERENCE_CYCLES_FALLBACK_NOT_OFFICIAL_TEST_NOT_FAIR_BENCHMARK','width':1024,'height':576,'frames':25,'seed':20,'obj_num':4,'focal_length_mm':50.,'sensor_width_mm':36.,'cycles_samples':1024,'source_commit':'f7a16959dd219080fb8eec0d32bb9602d8970c22','source_generator':'https://github.com/vivoCameraResearch/any-to-bokeh/blob/f7a16959dd219080fb8eec0d32bb9602d8970c22/dataset_synthsis.py','missing':['z_bokeh.data_process.bokeh_kernel.ModuleRenderRT','merge_f25_test.json','BG_test background list'],'differences':['200 base scenes and one fixed focus-object variant each; public script currently 300 bases with four focus variants','AM train FG treated as fixed asset, not physical color ground truth; no recovery','VM HEVC decoded at stride 4; max long side 1920 for JPEG-HD-like source size, not identical JPEG files','BG-20k train replaces undisclosed BG_test; official validation/test pools untouched','Seed 20 plus scene ID defines independent deterministic NumPy/Torch streams instead of process-dependent RNG','Public crop alpha>128, scale0.5..0.8, rotate-10..10, grid2x2 random-walk placements retained, including crop/border limitations','Cycles physical linear-light composition with matching linear RGB inputs replaces author sRGB AIF and unavailable ray renderer','PNG8 input/disparity and PNG16 GT instead of JPEG, rounded disparity; no per-scene normalization','Max blur radius can approach32px, beyond E6 quantitatively tested2.67..10.67px; do not inherit full E6 calibration claim'],'scenes':scenes}
    with path.open('x') as f:json.dump(cfg,f,indent=2)

def generate(cfg,scene,out):
    cv2.setNumThreads(1);torch.set_num_threads(1)
    rng=np.random.RandomState(scene['seed']);g=torch.Generator().manual_seed(scene['seed'])
    W,H=cfg['width'],cfg['height'];folder=out/scene['id'];folder.mkdir()
    for name in ['rgb','disparity','depth','alpha','mask','layers']:(folder/name).mkdir()
    bg=np.asarray(Image.open(ROOT/scene['background_file']).convert('RGB').resize((W,H)),dtype=np.uint8)
    Image.fromarray(np.pad(bg,((64,64),(64,64),(0,0)),mode='reflect')).save(folder/'layers/background.png')
    background=srgb_to_linear(bg.astype(np.float32)/255)
    centers=[]
    for row in range(2):
        for col in range(2):
            centers.append(np.array([int((col+.5)*(W-20)/2),int((row+.5)*(H-20)/2)])+np.array([rng.randint(-int((W-20)/4),int((W-20)/4)),rng.randint(-int((H-20)/4),int((H-20)/4))]))
    depths=(torch.linspace(1,5,4)+(torch.rand((25,4),generator=g)*.5-.25)).clamp(1,5).numpy()
    permutation=torch.randperm(4,generator=g).numpy();order=np.argsort(permutation)
    bgdepth=(torch.rand(25,generator=g)*100+100).numpy()
    assets=[];caps=[];transforms=[]
    for spec in scene['objects']:
        if spec['kind']=='AM':
            assets.append((Image.open(ROOT/spec['rgb']).convert('RGB'),Image.open(ROOT/spec['alpha']).convert('L')));caps.append(None)
        else:
            cs=[cv2.VideoCapture(str(ROOT/spec[k])) for k in ['rgb','alpha']]
            assert all(c.isOpened() for c in cs)
            for c in cs:c.set(cv2.CAP_PROP_POS_FRAMES,spec['start'])
            caps.append(cs);assets.append(None)
        transforms.append(None)
    c={k:cfg[k] for k in ['width','height','focal_length_mm','sensor_width_mm']};c.update(seed=scene['seed'],K=scene['K'],background_depth_m=float(bgdepth[0]))
    meta={'config':c,'source':scene,'role':'quick_reproduction_nonheldout_not_E7','geometry':'dynamic layered planar synthesis','color_space':'linear composition matching Cycles; differs from public AIF sRGB','frames':[]}
    try:
        for t in range(25):
            layers=[];placements=[];frame_cuts=[]
            for i,spec in enumerate(scene['objects']):
                if caps[i]:
                    decoded=[]
                    for cap in caps[i]:
                        ok,frame=cap.read();assert ok,(spec,t)
                        decoded.append(frame)
                        if t<24:
                            for _ in range(3):assert cap.grab()
                    color=Image.fromarray(cv2.cvtColor(decoded[0],cv2.COLOR_BGR2RGB));alpha=Image.fromarray(decoded[1][:,:,0])
                    if max(color.size)>1920:
                        size=tuple(round(v*1920/max(color.size)) for v in color.size)
                        color=color.resize(size);alpha=alpha.resize(size)
                else:color,alpha=assets[i]
                aa=np.asarray(alpha);yy,xx=np.where(aa>128);assert len(xx),(scene['id'],i,t,'empty alpha')
                frame_cuts.append(bool(aa[0].any() or aa[-1].any() or aa[:,0].any() or aa[:,-1].any()))
                box=(int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max()))
                color=color.crop(box);alpha=alpha.crop(box)
                if t==0:transforms[i]=(min(W/color.width,H/color.height)*rng.uniform(.5,.8),float(rng.uniform(-10,10)))
                scale,angle=transforms[i];size=(int(color.width*scale),int(color.height*scale));assert min(size)>0 and size[0]<=W and size[1]<=H
                color=color.resize(size).rotate(angle);alpha=alpha.resize(size).rotate(angle)
                centers[i]+=rng.randint(-40,40,2)
                x=int(np.clip(centers[i][0]-size[0]/2,0,W-size[0]));y=int(np.clip(centers[i][1]-size[1]/2,0,H-size[1]))
                rgba=np.zeros((H,W,4),np.uint8);rgba[y:y+size[1],x:x+size[0],:3]=np.asarray(color);rgba[y:y+size[1],x:x+size[0],3]=np.asarray(alpha)
                layers.append(rgba);placements.append({'crop_xyxy':box,'scale':scale,'angle':angle,'xywh':[x,y,*size]})
            rgb=background.copy();disp=np.full((H,W),1/bgdepth[t],np.float32);union=np.zeros((H,W),np.float32)
            files=[]
            for rank,i in enumerate(order):
                file=f'layers/{t:05d}_fg{rank}.png';Image.fromarray(layers[i]).save(folder/file);files.append(file)
            for rank in range(3,-1,-1):
                layer=layers[order[rank]].astype(np.float32)/255;aa=layer[:,:,3];cc=srgb_to_linear(layer[:,:,:3])
                rgb=cc*aa[...,None]+rgb*(1-aa[...,None]);disp=np.where(aa>.5,1/depths[t,rank],disp);union=aa+union*(1-aa)
            focus=float(depths[t,scene['focus_object_depth_rank']]);assert np.isfinite(rgb).all() and (disp>0).all() and (disp<=1).all()
            Image.fromarray(np.uint8(np.round(linear_to_srgb(rgb)*255))).save(folder/'rgb'/f'{t:05d}.png')
            Image.fromarray(np.uint8(np.round(disp*255))).save(folder/'disparity'/f'{t:05d}_zf_{1/focus:.8f}.png')
            Image.fromarray(np.uint8(np.round(union*255))).save(folder/'alpha'/f'{t:05d}.png')
            Image.fromarray(np.uint8(union>.5)*255).save(folder/'mask'/f'{t:05d}.png');np.save(folder/'depth'/f'{t:05d}.npy',1/disp)
            meta['frames'].append({'index':t,'timeline_frame':t,'focus_distance_m':focus,'focus_disparity':1/focus,'background_depth_m':float(bgdepth[t]),'foreground_files':files,'foreground_depths_m':depths[t].tolist(),'source_order_by_depth':order.tolist(),'placements_by_source':placements,'source_boundary_alpha_nonzero':frame_cuts})
    finally:
        for pair in caps:
            if pair:
                for cap in pair:cap.release()
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2));return scene['id']

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/any2bokeh-reproduction-200.json');p.add_argument('--build',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--only',default='');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.build:build_config(a.config);return
    cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
    cp=a.output/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:cp.write_text(json.dumps(cfg,indent=2));(a.output/'render_settings.json').write_text(json.dumps({'fstop_scale':1.}))
    scenes=[s for s in cfg['scenes'] if not a.only or s['id'] in a.only.split(',')]
    assert scenes
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(generate,cfg,s,a.output) for s in scenes]
        for i,f in enumerate(as_completed(futures)):print(i+1,len(scenes),f.result(),flush=True)
if __name__=='__main__':main()
