"""Create reproducible shared RGB/geometry/layer inputs; raw assets stay untouched."""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from synthetic.common import srgb_to_linear, linear_to_srgb
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=root/'configs/renderer-pilot.json');a=p.parse_args()
cfg=json.loads(a.config.read_text());W,H=cfg['width'],cfg['height'];out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
fgs=[root/x for x in cfg['foreground_files']] if 'foreground_files' in cfg else sorted((root/'data/raw/AM-2k/train/fg').glob('*.png'))[:5]
bgs=[root/x for x in cfg['background_files']] if 'background_files' in cfg else sorted((root/'data/raw/BG-20k/testval').glob('*.jpg'))[:5]
assert len(fgs)==len(bgs)==5

def save_rgb(path,linear): Image.fromarray(np.uint8(np.round(linear_to_srgb(linear)*255))).save(path)

def make_scene(si,calibration=False):
    folder=out/('calibration' if calibration else f'scene_{si:03d}')
    for n in ['rgb','disparity','depth','alpha','mask','layers']:(folder/n).mkdir(parents=True)
    if calibration:
        bg=np.zeros((H,W,3),np.float32);bg[:,W//2:]=1
    else:
        im=np.array(Image.open(bgs[si]).convert('RGB'))
        h,w=im.shape[:2];scale=max(W/w,H/h);im=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
        y=(im.shape[0]-H)//2;x=(im.shape[1]-W)//2
        bg=srgb_to_linear(im[y:y+H,x:x+W].astype(np.float32)/255)
    save_rgb(folder/'layers/background.png',np.pad(bg,((64,64),(64,64),(0,0)),mode='reflect'))
    layer_assets=[]
    if not calibration:
        for li in range(2 if si in [2,4] else 1):
            fg=fgs[(si+li)%len(fgs)];mask=root/'data/raw/AM-2k/train/mask'/fg.name
            color=srgb_to_linear(np.array(Image.open(fg).convert('RGB')).astype(np.float32)/255)
            alpha=np.array(Image.open(mask).convert('L')).astype(np.float32)/255
            h,w=alpha.shape;scale=min(420/h,650/w)*(0.8 if li else 1)
            layer_assets.append((color,alpha,scale,fg,mask,cfg['foreground_depth_m']+li))
    meta={'config':cfg,'role':'calibration' if calibration else 'renderer_development_only','scene':si,'background_source':None if calibration else str(bgs[si].relative_to(root)),'frames':[],'color_space':'linear-light composition and rendering, sRGB PNG delivery','disparity_rule':'inverse meters; frontmost alpha > 0.5 selects geometry; alpha stored separately','background_padding_pixels':64}
    for t in range(1 if calibration else cfg['frames']):
        layers=[];transforms=[]
        for li,(color,alpha,scale,fg,mask,z) in enumerate(layer_assets):
            angle=3*np.sin(t/max(cfg['frames']-1,1)*2*np.pi+si)
            h,w=alpha.shape;M=cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
            center=np.array([W*(.43+.13*li)+35*np.sin(t/max(cfg['frames']-1,1)*2*np.pi+si),H*.51+12*np.cos(t/max(cfg['frames']-1,1)*2*np.pi+si)])
            M[:,2]+=center-np.array([w/2,h/2])
            transforms.append({'layer':li,'depth_m':z,'affine_source_to_canvas':M.tolist()})
            aa=cv2.warpAffine(alpha,M,(W,H),flags=cv2.INTER_LINEAR)
            premul=cv2.warpAffine(color*alpha[...,None],M,(W,H),flags=cv2.INTER_LINEAR)
            cc=np.divide(premul,aa[...,None],out=np.zeros_like(premul),where=aa[...,None]>1e-6)
            png=folder/'layers'/f'{t:05d}_fg{li}.png'
            rgba=np.dstack((linear_to_srgb(cc),aa));Image.fromarray(np.uint8(np.round(np.clip(rgba,0,1)*255)),'RGBA').save(png)
            # Re-read delivered textures to make CPU and Cycles use identical quantized assets.
            q=np.array(Image.open(png)).astype(np.float32)/255
            layers.append((srgb_to_linear(q[:,:,:3]),q[:,:,3],z))
        bg_file=np.array(Image.open(folder/'layers/background.png')).astype(np.float32)/255
        comp=srgb_to_linear(bg_file[64:64+H,64:64+W]);disp=np.full((H,W),1/cfg['background_depth_m'],np.float32);union=np.zeros((H,W),np.float32)
        for cc,aa,z in reversed(layers):
            comp=cc*aa[...,None]+comp*(1-aa[...,None]);disp=np.where(aa>.5,1/z,disp);union=aa+union*(1-aa)
        focus=1/cfg['background_depth_m'] if si==3 and not calibration else 1/cfg['foreground_depth_m']
        save_rgb(folder/'rgb'/f'{t:05d}.png',comp)
        Image.fromarray(np.uint8(np.round(disp*255))).save(folder/'disparity'/f'{t:05d}_zf_{focus:.8f}.png')
        Image.fromarray(np.uint8(np.round(union*255))).save(folder/'alpha'/f'{t:05d}.png')
        Image.fromarray(np.uint8(union>.5)*255).save(folder/'mask'/f'{t:05d}.png')
        np.save(folder/'depth'/f'{t:05d}.npy',1/disp)
        meta['frames'].append({'index':t,'focus_disparity':focus,'focus_distance_m':1/focus,'foreground_depths_m':[x[2] for x in layers],'foreground_files':[f'layers/{t:05d}_fg{li}.png' for li in range(len(layers))],'transforms':transforms,'layer_order':'front_to_back','focus_target':'background' if si==3 and not calibration else 'foreground_0'})
    meta['foreground_sources']=[{'fg':str(x[3].relative_to(root)),'alpha':str(x[4].relative_to(root))} for x in layer_assets]
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(folder,flush=True)
make_scene(0,True)
for si in range(cfg['scenes']):make_scene(si)
(out/'config.json').write_text(json.dumps(cfg,indent=2))
