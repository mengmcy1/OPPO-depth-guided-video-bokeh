import numpy as np, OpenEXR
from PIL import Image
from matplotlib import cm
O='/root/autodl-tmp/OPPO-depth-guided-video-bokeh/outputs/'
T='/tmp/claude-0/-root-autodl-tmp/ac32074c-db8a-4e0f-87bd-8cdb40db961a/scratchpad/build/tiles/'
def srgb(l):
    l=np.clip(l,0,1); return np.where(l<=0.0031308,12.92*l,1.055*np.power(l,1/2.4)-0.055)
def gt(case):
    f={'f001-N1.4-F1.4':'f001-N1.4-F1.4-s2048-seed20261110-hidenone.exr','f030-N1.4-F5':'f030-N1.4-F5-s1024-seed20261110-hidenone.exr'}[case]
    with OpenEXR.File(O+'route3-b2only-2026-11-10-01/ground-truth/B2-high/'+f) as e: return srgb(e.channels()['RGB'].pixels)
def save(a,name): Image.fromarray((np.clip(a,0,1)*255+0.5).astype(np.uint8)).save(T+name+'.png')
def errmap(p,g,vmax=0.04):
    e=np.abs(p-g).mean(axis=2)/vmax
    return cm.inferno(np.clip(e,0,1))[...,:3]
for case in ['f001-N1.4-F1.4','f030-N1.4-F5']:
    g=gt(case); save(g,f'p9e_{case}_gt')
    preds={'o3':np.load(O+f'route3-sharp-s1-2026-11-11-01/O3/native-B2-high-{case}.npz')['prediction_srgb'],
           'sharp':np.load(O+f'route3-sharp-s1-2026-11-11-01/render/native-B2-high-{case}-SHARP_full.npz')['prediction_srgb'],
           'anchor':np.load(O+f'route3-sharp-s1b-2026-11-11-01/render-native/B2-high-{case}-A_color.npz')['prediction_srgb']}
    for k,p in preds.items():
        save(p,f'p9e_{case}_{k}'); save(errmap(p,g),f'p9e_{case}_{k}_err')
        e=np.abs(p-g); print(case,k,'MAE %.4f'%e.mean(), 'PSNR %.2f'%(10*np.log10(1/((p-g)**2).mean())))
# transmittance maps (1-alpha) VG vs SHARP_full, f001 F1.4 N1.4
for k in ['VG_SHARP','SHARP_full']:
    a=np.load(O+f'route3-sharp-s1-2026-11-11-01/render/native-B2-high-f001-N1.4-F1.4-{k}.npz')['alpha']
    t=np.clip((1-a)/0.3,0,1); save(cm.inferno(t)[...,:3],f'p4_trans_{k}'); print(k,'leaky>0.05 frac %.3f'%((1-a)>0.05).mean())
