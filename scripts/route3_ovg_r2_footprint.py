"""Correct the W4 guard; compare both rules on the same trained Query."""

import numpy as np
import torch

from route3_ovg_r2_common import CONFIG, LOG, OUT, ROUND1, guard, pooled_psnr
import route3_ovg_footprint as w4
from route3_ovg_subpixel import save_json, save_npz
from src.dual_layer import depth_band
from src.layered_aperture import merge_layers
from src.ordered_layers import order_hidden, ray_samples, render_fallbacks


@torch.inference_mode()
def evaluate(dest):
    net = w4.query_net(dest/'retrained/step-020000.pt')
    store = w4.Store()
    metric = w4.lpips()
    rows = []
    for row in w4.read(dest/'decision.json')['rows']:
        guard()
        name, cid = row['scene'], row['case']
        _, d, A, D, K, z, fill, bg, a_zero, rgb = w4.context(name)
        source = w4.PREVIOUS/'evaluation-input/original'/name/'512-13/manifest.json'
        manifest = w4.read(source)
        case = next(x for x in manifest['rows'] if x['case']['id']==cid and x['case']['protocol']=='P2')
        c = case['case']
        focus, radius = c['focus_m'], .05/(2*c['fstop'])
        band = depth_band(d['depth'],d['K'][0,0],focus,radius)[0]
        hidden = w4.learned_hidden(net,store,case,manifest,K,band)
        center, center_stat = order_hidden(hidden,z,K,'O1')
        rules = [
            ('center_O1', center, center_stat),
            ('footprint', *w4.apply_footprint_order(hidden,z,K,w4.CONFIG['footprint_min_sigma_px_512'])),
        ]
        raw_a, alpha_a = w4.zero_parts(A,K)
        a_reference = w4.display((raw_a+(1-alpha_a[...,None])*fill).cpu().numpy())
        a_mse = np.mean((a_reference.astype(np.float64)-rgb)**2)
        a_zero_db = float(-10*np.log10(max(a_mse,1e-30)))
        reference = w4.PREVIOUS/'truth'/name
        gt = w4.display(w4.read_exr(reference/'dof'/(cid+'.exr')))
        regions = w4.masks(reference,dict(c,frame=30))
        samples = ray_samples(A,K,focus,radius,64,fill,bg)
        for method, cloud, audit in rules:
            coupled=merge_layers(D,cloud)
            pred=w4.display(render_fallbacks(coupled,K,focus,radius,64,fill,bg,samples)['F2'].cpu().numpy())
            measured=w4.score(pred,gt,{k:regions[k] for k in ['recoverable','band','in_focus','full_image']},metric)
            zv=w4.zero(coupled,K,a_zero,rgb)
            rows.append(dict(scene=name,case=cid,nominal_focus=c['nominal_focus_m'],
                             method=method,zero_PSNR=zv,zero_vs_A=zv-a_zero_db,
                             below_absolute_60=zv<CONFIG['zero_absolute_min_db'],zero_pixels=rgb.shape[0]*rgb.shape[1],
                             audit=audit,regions=measured))
            save_npz(dest/'predictions'/name/(cid+'-'+method+'.npz'),prediction_srgb=pred)
    combined = {}
    for method in ('center_O1','footprint'):
        g = [r for r in rows if r['method']==method]
        regions = {}
        for key in g[0]['regions']:
            n=sum(h['regions'][key]['pixels'] for h in g)
            s=sum(h['regions'][key]['sse'] for h in g)
            regions[key] = dict(pixels=n, PSNR=float(-10*np.log10(max(s/(3*n),1e-30))) if n else None,
                                LPIPS=sum(h['regions'][key]['patch_lpips']*h['regions'][key]['pixels']
                                          for h in g if h['regions'][key]['pixels'])/n if n else None)
        combined[method]=dict(zero_PSNR=pooled_psnr([(r['zero_pixels'],r['zero_PSNR']) for r in g]),regions=regions)
    paired = {}
    for r in rows:
        paired.setdefault((r['scene'],r['case']),{})[r['method']]=r['zero_PSNR']
    winner=max(combined,key=lambda k:combined[k]['zero_PSNR'])
    save_json(dest/'evaluation.json',dict(rows=rows,combined=combined,measured_rule=winner,
              footprint_lost_configs=[dict(scene=a,case=b) for (a,b),p in paired.items() if p['footprint']<p['center_O1']],
              decision_basis='combined MSE in interior b=0; absolute 60 dB and relative A reported separately',
              renders='both rules use the same footprint-trained Query, same O1 center and F2 background'))


def main():
    guard()
    dest=OUT/'V1'
    dest.mkdir(exist_ok=True)
    w4.WORK,w4.OUT,w4.LOG = dest, OUT, LOG
    before=w4.read(ROUND1/'W4/decision.json')
    # The previous paired render used the same center and footprint
    # configurations. Reinterpret its zero-aperture measurements only.
    combined={k:pooled_psnr([(512**2,r[k+'_zero_PSNR']) for r in before['rows']])
              for k in ('center','footprint')}
    if combined['footprint']<combined['center']:
        raise RuntimeError('Stored guard results do not support the protocol choice')
    save_json(dest/'decision.json',dict(selected='footprint',rows=before['rows'],previous_pooled_zero=combined,
              absolute_floor_db=CONFIG['zero_absolute_min_db'],old_D0_comparison_used=False))
    torch.set_num_threads(3)
    train,split=w4.make_training('footprint')
    w4.guard(True)
    w4.train_network(train,dest/'retrained',split['train'],split['heldout'],require_descent=True)
    evaluate(dest)


if __name__=='__main__':
    main()
