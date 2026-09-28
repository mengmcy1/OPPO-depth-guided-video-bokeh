"""Check nominal CoC units and finite-window effects, without running a baseline sweep."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/drbokeh-scale-probe-20260928'))
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    protocol = json.loads((ROOT/'configs/drbokeh-track-a-preflight.json').read_text())
    repo = ROOT.parent/'DrBokeh'
    head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert head == protocol['author_commit']
    sys.path.insert(0,str(repo/'app/Render/DScatter'))
    from scatter import Multi_Layer_Renderer
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe = protocol['kernel_check']
    n = probe['impulse_size']
    y,x = np.mgrid[:n,:n].astype(np.float64)-n//2
    rho = np.sqrt(x*x+y*y)
    # 257x257 includes all measurable tails for the tested <=32.1 px nominal radii.
    grid = np.arange(-128,129,dtype=np.float64)
    rho_large = np.hypot(grid[:,None],grid[None,:])
    renderers = {lens:Multi_Layer_Renderer(lens).cuda().eval() for lens in probe['lens_candidates_for_coverage_check_only']}
    records = []
    with torch.no_grad():
        for radius in probe['nominal_radii_px']:
            plane = np.zeros((n,n,5),dtype=np.float32)
            plane[n//2,n//2,:3] = 1
            plane[...,3] = 1
            plane[...,4] = radius/32
            outputs,measurements = {},{}
            # Gamma is intentionally absent: this measures the renderer's linear impulse kernel.
            for lens,renderer in renderers.items():
                value = renderer.inference([plane],32,0)[...,0]
                assert value.shape==(n,n) and np.isfinite(value).all() and value.min()>=0
                outputs[lens] = value
                mass = float(value.sum())
                assert mass > 0
                effective = float(np.sqrt(2*np.sum(value*rho**2)/mass))
                weight = .1/(.1+np.exp(-3*((radius+1)-rho_large))+1e-6)/(radius+1)**2
                excluded = float(weight[rho_large>=lens//2].sum()/weight.sum())
                measurements[str(lens)] = dict(impulse_mass=mass,second_moment_equivalent_disk_radius=effective,
                                               central_value=float(value[n//2,n//2]),omitted_radial_weight_mass=excluded)
            diffs = {str(l):dict(L1=float(np.abs(outputs[l]-outputs[101]).sum()),
                                  max_abs=float(np.abs(outputs[l]-outputs[101]).max())) for l in (41,71)}
            reverse = plane.copy();reverse[...,4]=0
            reverse_value = renderers[71].inference([reverse],32,radius/32)[...,0]
            sign_error = float(np.max(np.abs(reverse_value-outputs[71])))
            assert sign_error <= 1e-6
            row = dict(nominal_radius=radius,kernel_scatter_dis=radius+1,lenses=measurements,
                       difference_to_lens101=diffs,positive_negative_defocus_max_abs=sign_error,
                       lens71_passed=bool(diffs['71']['L1']<=probe['lens71_vs101_impulse_L1_tolerance'] and measurements['71']['omitted_radial_weight_mass']<=probe['omitted_radial_scatter_weight_mass_tolerance']))
            records.append(row)
            with (args.output/f'impulse_R{radius:g}.npz').open('xb') as f:
                np.savez_compressed(f,**{f'lens_{l}':a for l,a in outputs.items()})
            print('RADIUS',radius,'lens71 vs101',diffs['71'],'pass',row['lens71_passed'],flush=True)
    clips = ROOT/protocol['input']
    scenes = {s['id']:s for s in json.loads((clips/'config.json').read_text())['scenes']}
    inputs = []
    for sid in protocol['preflight_scenes']:
        meta = json.loads((clips/sid/'metadata.json').read_text())
        depths = sorted((clips/sid/'disparity').glob('*.png'));assert len(depths)==25
        for t,path in enumerate(depths):
            d = np.asarray(Image.open(path)).astype(np.float64)/255
            assert d.shape == (576,1024)
            f = float(path.stem.split('_zf_')[1]);assert abs(f-meta['frames'][t]['focus_disparity'])<=5.1e-9
            lo,hi = float(d.min()),float(d.max());span=hi-lo;assert span>0
            k=scenes[sid]['K'];dn=(d-lo)/span;fn=(f-lo)/span;kn=k*span
            error=float(np.abs(kn*(dn-fn)-k*(d-f)).max());assert error<1e-10
            inputs.append(dict(scene=sid,frame=t,K_manifest=k,K_Dr=kn,span=span,focus_prime=fn,
                               nominal_radius_max=float(np.max(np.abs(k*(d-f)))),mapping_max_abs=error))
    report=dict(status='KERNEL_AND_INPUT_SCALE_CHECK_COMPLETE_NOT_FULL_PREFLIGHT',author_commit=head,
                protocol=protocol,impulse_results=records,input_frames=inputs,
                all_lens71_tests_passed=all(x['lens71_passed'] for x in records),
                max_nominal_input_radius=max(x['nominal_radius_max'] for x in inputs),
                note='No GT alpha/layers or paper metrics consumed. Constant-plane kernels are analytic controls only. Post-inpainting layers still require the full 8-scene preflight.')
    with (args.output/'result.json').open('x') as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print('COMPLETE',report['all_lens71_tests_passed'],'max input radius',report['max_nominal_input_radius'],flush=True)


if __name__ == '__main__':
    main()
