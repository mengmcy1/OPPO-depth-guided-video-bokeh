"""C0 first: audit saved MoVieS source order without another model forward.

Compare real head colors to aligned input pixels, match all exported color and
opacity rows, and send (frame,row,column) marker colors through the official
renderer. Marker colors are diagnostics only, never method predictions.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--repository',type=Path,default=Path('/root/autodl-tmp/MoVieS'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();a.project=a.project.resolve();a.repository=a.repository.resolve();a.output=a.output.resolve();a.record=a.record.resolve()
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('camvfx_input',a.project/'src/camvfx4d.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    os.chdir(a.repository);sys.path[:0]=[str(a.repository),str(a.repository/'extensions/vggt')]
    from src.options import opt_dict
    from src.models.gs_render.gs_renderer import GaussianRenderer
    results=[]
    for name,relative in [('box','outputs/route3-prep-2026-11-09/movies-box-03'),
                          ('plant','outputs/route3-c1-2026-11-10-01/movies-plant-01'),
                          ('cat','outputs/route3-c1-2026-11-10-01/movies-cat-01')]:
        cache=a.project/relative;raw=a.project/'data/raw/CamVFX-4D'/name
        saved=torch.load(cache/'same_forward_outputs.pt',map_location='cpu',weights_only=True)
        exported=np.load(cache/'metric_gaussians.npz');ids=saved['input_frame_ids'];F=len(ids);H,W=saved['color'].shape[-2:]
        motion=saved['query_motion_gaussians'][0]
        opacity=(torch.tanh(motion['motion_opacity'][0].permute(0,2,3,1).reshape(-1)-1)*.5+.5)
        flat=(opacity>.001).nonzero().flatten().numpy()
        expected_sources=(np.array(ids)-1)[flat//(H*W)]
        source_mismatches=int(np.count_nonzero(expected_sources!=exported['source_frames']))
        flat_mismatches=int(np.count_nonzero(flat!=exported['source_flat_indices'])) if 'source_flat_indices' in exported else None
        colors=(torch.tanh(motion['motion_color'])*.5+.5)[0].permute(0,2,3,1).reshape(-1,3).numpy()[flat]
        color_error=float(np.max(np.abs(colors-exported['colors_srgb'])))
        opacity_error=float(np.max(np.abs(opacity.numpy()[flat]-exported['opacities'])))
        frames,metadata=helper.load_camera_frames(raw/'pinhole/transforms_train.json')
        images=np.stack([cv2.resize(helper.read_rgb(raw/f'pinhole/0/{i:06d}.png'),(W,H),interpolation=cv2.INTER_CUBIC).clip(0,1) for i in ids])
        if name=='box':
            from PIL import Image
            images=np.stack([np.asarray(Image.open(raw/f'pinhole/0/{i:06d}.png').convert('RGB').resize((W,H),Image.Resampling.BICUBIC),np.float32)/255 for i in ids])
        images=torch.from_numpy(images).to(torch.bfloat16).float().numpy()
        base=(torch.tanh(saved['color'])*.5+.5)[0].permute(0,2,3,1).numpy()
        temporal=np.max(images.max(0)-images.min(0),axis=-1)>.03
        similarities=[]
        for f in range(F):
            scores=[float(np.abs(base[f]-images[k])[temporal].mean()) if temporal.any() else float(np.abs(base[f]-images[k]).mean()) for k in range(F)]
            similarities.append(dict(source_frame_id=ids[f],mae_against_input_frames_on_temporal_pixels=scores,
                                     closest_frame_id=ids[int(np.argmin(scores))],assigned_frame_mae=scores[f]))
        options=copy.deepcopy(opt_dict['movies']);options.color_act_type='identity'
        assert options.voxel_size==0,'Voxel aggregation needs explicit provenance handling'
        renderer=GaussianRenderer(options)
        inputs={k:saved[k].to('cuda') for k in ['depth','color','scale','rotation','opacity','conf']}
        inputs['offset']=saved['query_motion'][:,0,:,:3].to('cuda')
        inputs.update({k:v.to('cuda') for k,v in motion.items()})
        # Identity activation lets this track every surviving source pixel.
        f,y,x=torch.meshgrid(torch.arange(F,device='cuda'),torch.arange(H,device='cuda'),torch.arange(W,device='cuda'),indexing='ij')
        marker=torch.stack([f/max(F-1,1),y/max(H-1,1),x/max(W-1,1)],1)[None].float()
        inputs['color']=marker;inputs['motion_color']=marker
        cameras=saved['normalized_C2W'][None].to('cuda',dtype=torch.bfloat16)
        K=helper.intrinsics_from_fov(metadata,W,H)
        intrinsics=torch.tensor([K[0,0]/W,K[1,1]/H,.5,.5],device='cuda',dtype=torch.bfloat16)[None,None].expand(1,F,4)
        with torch.inference_mode():result=renderer.render(inputs,cameras,intrinsics,cameras[:,:1],intrinsics[:,:1],return_pc=True)
        pc=result['pc'][0]
        expected=marker[0].permute(0,2,3,1).reshape(-1,3)[torch.tensor(flat,device='cuda')]
        marker_error=float((pc.color-expected).abs().max())
        xyz_error=float(np.max(np.abs(pc.xyz.cpu().numpy()*float(exported['scale_to_meters'])-exported['means'])))
        row=dict(scene=name,cache=str(cache),gaussians=len(pc.xyz),input_frame_ids=ids,
            source_id_mismatches=source_mismatches,saved_flat_index_mismatches=flat_mismatches,
            activated_query_color_export_max_error=color_error,activated_query_opacity_export_max_error=opacity_error,
            official_marker_frame_row_column_max_error=marker_error,official_xyz_export_max_error_m=xyz_error,
            source_pixel_rgb_similarity=similarities,temporal_rgb_pixels=int(temporal.sum()),
            RGB_shortcut_note='Official shortcut is learned convolutions, not an exact identity RGB copy; marker/row matching is decisive for numbering',
            ordering='B,F,H,W flattened, channels last; native boolean opacity pruning preserves order',
            passed=source_mismatches==0 and flat_mismatches in [None,0] and marker_error<1e-7 and color_error<1e-5 and opacity_error<1e-5 and xyz_error<1e-4)
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='source_pixel_rgb_similarity'}),flush=True)
        del inputs,result,pc,expected,marker,exported,saved;torch.cuda.empty_cache()
    record=dict(phase='C0 source numbering first',official_commit='77262fac6724a12c0b133a4fee10e62772db6d0f',results=results,
                source_numbering_error_found=not all(r['passed'] for r in results),
                A1_original_outputs_modified=False,test_method_evaluation=False,new_model_forward=False,
                conclusion='No source numbering error found' if all(r['passed'] for r in results) else 'Stop and report affected source indexing before A/B')
    a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    (a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    if record['source_numbering_error_found']:raise SystemExit(2)


if __name__=='__main__':main()
