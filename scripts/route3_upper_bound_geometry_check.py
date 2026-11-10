"""U0 pilot overlay: exact same off-axis K/E in Blender and native gsplat."""
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
from gsplat import rasterization
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.layered_aperture import pixel_layer,merge_layers,aperture_fill_samples
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_upper_bound_evaluate import display,exr


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True)
    torch.set_num_threads(4)
    out=ROOT/'outputs/route3-upper-bound-2026-11-11-01'
    dest=out/'U0-geometry-overlay';dest.mkdir(exist_ok=False)
    rows=json.loads((out/'U0-subviews/timings.json').read_text())
    c=json.loads((ROOT/'03_实验记录/records/route3-dual-layer-criteria-L2-2026-11-11.json').read_text());results=[]
    for r in rows:
        frame=r['frame'];name=f'B2-high-f{frame:03d}'
        d=np.load(ROOT/'outputs/route3-dual-layer-2026-11-10-01/shared-inputs'/(name+'.npz'))
        K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32)
        visible=pixel_layer(torch.tensor(d['visible_rgb_linear'],device='cuda'),torch.tensor(d['center_depth'],device='cuda'),K,cam)
        parent=np.load(ROOT/'outputs/route3-b2only-2026-11-10-01/methods-3d/prepared'/name/'parents.npz')
        hidden=sparse_pixel_layer(parent,parent['recoverable'],K,cam,c['profile'],'cuda');pc=merge_layers(visible,hidden)
        intr=torch.tensor(r['K'],device='cuda',dtype=torch.float32);view=torch.linalg.inv(torch.tensor(r['camera_c2w'],device='cuda',dtype=torch.float32))
        with torch.inference_mode():
            rgb,alpha,_=rasterization(pc.means,pc.quats,pc.scales,pc.opacities,pc.colors,view[None],intr[None],512,512,
                packed=False,rasterize_mode='antialiased',eps2d=.0015,near_plane=1e-4,far_plane=1e5)
        target=display(exr(r['path']));gs=display(rgb[0].cpu().numpy())
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,im,title in zip(axes,[target,gs,.5*(target+gs)],['Blender same ray','GT-geometry GS (appearance/coverage approximate)','50/50 overlay']):
            ax.imshow(im);ax.set_title(title);ax.axis('off')
        fig.tight_layout();path=dest/(r['key']+'.png');fig.savefig(path,dpi=120);plt.close(fig)
        # Independent check: translated camera in original coordinates and K.
        E=torch.linalg.inv(cam);E[:2,3]-=torch.tensor(r['xy_m'],device='cuda',dtype=torch.float32)
        expected=K.clone();expected[:2,2]+=torch.diag(K)[:2]*torch.tensor(r['xy_m'],device='cuda')/r['focus_m']
        check=dict(key=r['key'],view_matrix_max_abs_error=float(abs(E-view).max()),
            intrinsic_matrix_max_abs_error=float(abs(expected-intr).max()),projection_error_px=r['projection_error_px'],
            image=str(path),appearance_is_not_protocol_geometry_test=True)
        if check['view_matrix_max_abs_error']>1e-5 or check['intrinsic_matrix_max_abs_error']>1e-4:raise ValueError(check)
        results.append(check);del pc,visible,hidden,rgb,alpha;torch.cuda.empty_cache()
    geometry=json.loads((out/'U0-subviews/geometry-checks.json').read_text())
    report=dict(passed=True,rows=results,all768_projection_error_max_px=max(r['projection_error_px'] for r in geometry['rows']),
        method='Blender actual projection and project off-axis camera agree; same-subcamera overlay attached',
        no_Gaussian_appearance_fidelity_claim=True)
    (ROOT/'03_实验记录/records/route3-upper-bound-2026-11-11-U0-geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
