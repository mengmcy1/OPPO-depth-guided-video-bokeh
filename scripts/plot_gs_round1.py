#!/usr/bin/env python3
"""Export standalone E4 figures from completed, measured benchmark records."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
log=root/'logs/dof-gs-repro-2026-11-05-round1'
out=root/'outputs/dof-gs-repro-2026-11-05/figures';out.mkdir(parents=True,exist_ok=True)
names={'dofgs':'DOF-GS','dofgaussian':'DoF-Gaussian','cocogaussian':'CoCoGaussian'}
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for row,scene in enumerate(['cake','caps']):
    for method,label in names.items():
        path=log/('benchmark-'+method+'-'+scene+'.json')
        if not path.exists():continue
        data=json.loads(path.read_text())
        if data.get('status')!='benchmarked':continue
        points=[r for r in data['results'] if r['case'].startswith('e4_')]
        points.sort(key=lambda r:r['max_radius_px'])
        for col,key,scale in [(0,'milliseconds',1),(1,'torch_peak_bytes',1024**3)]:
            axes[row,col].plot([p['max_radius_px'] for p in points],[p[key]/scale for p in points],marker='o',label=label)
    for col in [0,1]:
        axes[row,col].set_title(scene);axes[row,col].set_xlabel('Maximum circle radius (pixels)')
        axes[row,col].set_xticks([0,2,4,8,16,32]);axes[row,col].grid(alpha=.25)
        axes[row,col].set_ylabel('Render time (ms)' if col==0 else 'PyTorch peak memory (GiB)')
        if axes[row,col].lines:axes[row,col].legend()
fig.suptitle('Measured E4: GPU0 exclusive; one view per scene; 50 repetitions\nCoCoGaussian uses projected learned position offsets')
fig.savefig(out/'e4-cost-curves.png',dpi=220)
fig.savefig(out/'e4-cost-curves.svg')
plt.close(fig)
