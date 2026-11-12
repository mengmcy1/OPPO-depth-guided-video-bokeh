"""Input-only verified observation index; GT is read only after index is saved."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_pilot_fields import OUT,read,write
from route3_pilot_candidates import load_views
from src.dense_flow_geometry import sample
from src.observation_memory import build_memory
from src.candidate_query_data import observation_flat_index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);a=p.parse_args();folder=OUT/a.scene
    with np.load(folder/'memory/observations.npz') as archive:raw={k:archive[k] for k in archive.files}
    views=load_views(folder);byframe={v['frame']:v for v in views};plan=json.loads((folder/'models/calibration/flow-plan.json').read_text())['selected'];yy,xx=np.mgrid[:512,:512];uv=np.stack([xx,yy],-1);verified=[];stats=[]
    for v in views:
        f=v['frame'];world=raw['position_world_estimated'][(v['frame']==raw['source_reference'][:,0])].reshape(512,512,3);count=np.zeros((512,512),np.uint8)
        for g in plan[str(f)]:
            other=byframe[g];flow=np.load(folder/'models/calibration/flow'/f'f{f:03d}-f{g:03d}.npz')['flow'];back=np.load(folder/'models/calibration/flow'/f'f{g:03d}-f{f:03d}.npz')['flow'];q=uv+flow;fb=np.linalg.norm(flow+sample(back,q),axis=-1);cp=(world-other['c2w'][:3,3])@other['c2w'][:3,:3];p=cp@v['K'].T;project=p[...,:2]/np.maximum(p[...,2:],1e-8);error=np.linalg.norm(project-(q+.5),axis=-1);count+=((error<=1)&(fb<=1)&(cp[...,2]>0)&(q[...,0]>=0)&(q[...,0]<=511)&(q[...,1]>=0)&(q[...,1]<=511)).astype(np.uint8)
        good=count>=2;verified.append(good.ravel());stats.append(dict(frame=f,verified_pixels=int(good.sum()),verified_fraction=float(good.mean())));print('PILOT_MEMORY_VERIFY',folder.name,f,good.sum(),flush=True)
    verified=np.concatenate(verified);path=folder/'memory/verified-index.npz'
    if not path.exists():
        m,summary=build_memory(raw['position_world_estimated'],raw['colors_linear'].astype(np.float32),raw['confidence'],raw['source_reference'],verified=verified);np.savez_compressed(path,**m);np.savez_compressed(folder/'memory/verification.npz',verified=verified);write(folder/'memory/verified-summary.json',dict(**summary,rows=stats,rule='estimated point must reproject to cachedinputflowwithin1px withFB<=1px,positivez,and>=2otherinputs;noGTdepth/IDs',GT_used=False))
    else:m=np.load(path)
    # Add a companion candidate field; all raw candidates remain traceable.
    for e in json.loads((folder/'prepared.json').read_text())['rows']:
        pool=np.load(e['pool']);index=observation_flat_index(pool['slot'],pool['source_xy']);p=Path(e['pool']).with_name('memory-evidence.npz')
        if not p.exists():np.savez_compressed(p,verified=verified[index],memory_entry_id=m['observation_entry_id'][index])
    # GT evaluation only starts here, after the input-only index is on disk.
    ids=[];world=[]
    for v in views:
        f=v['frame'];base=folder/'input-passes-v2';ids.append(read(base/f'f{f:03d}-IndexOB-{f:04d}.exr')[...,0].round().astype(np.int16).ravel());world.append(read(base/f'f{f:03d}-Position-{f:04d}.exr').reshape(-1,3))
    ids=np.concatenate(ids);world=np.concatenate(world);entry=m['observation_entry_id'];metadata=json.loads((folder/'scene.json').read_text());fg=[v['id'] for v in metadata['objects'] if v['name'] in metadata['foreground_names']];valid=(ids>0)&(entry>=0);ne=len(m['position']);isfg=np.isin(ids,fg);nf=np.bincount(entry[valid&isfg],minlength=ne);nb=np.bincount(entry[valid&~isfg],minlength=ne);mixed=(nf>0)&(nb>0)
    cells=np.c_[ids[valid],np.floor(world[valid]/.02).astype(np.int32)];unique,cellid=np.unique(cells,axis=0,return_inverse=True);pairs,counts=np.unique(np.c_[entry[valid],cellid],axis=0,return_counts=True);order=np.lexsort((-counts,pairs[:,0]));pairs=pairs[order];first=np.r_[True,np.diff(pairs[:,0])!=0];dominant=pairs[first];nunique=len(np.unique(dominant[:,1]));_,nentries=np.unique(np.unique(np.c_[cellid,entry[valid]],axis=0)[:,0],return_counts=True)
    report=dict(verified_entries=ne,entries_with_GT_support=len(dominant),foreground_background_wrong_merge_entries=int(mixed.sum()),foreground_background_wrong_merge_fraction=float(mixed.mean()),GT_surface_cells=len(unique),GT_cells_with_multiple_entries_fraction=float((nentries>1).mean()),dominant_GT_cell_redundant_entry_fraction=float(1-nunique/max(len(dominant),1)),entries_per_GT_cell_p50_p90_p99=np.quantile(nentries,[.5,.9,.99]).tolist(),surface_cell_definition='same object+2cmGTworldvoxel;resolution-dependent repetitionproxy,notexactcontinuous-surface duplicate',GT_only_diagnostic=True,original_unverified_index_excluded_from_long_term_memory=True)
    write(folder/'memory/GT-evaluation.json',report)


if __name__=='__main__':main()
