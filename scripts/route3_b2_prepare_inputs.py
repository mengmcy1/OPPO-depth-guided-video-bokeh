"""CPU-only shared LDR frame inputs and lens manifests before B2 methods.

No neural inference or method scoring. Source EXR remains scene-linear and
unclipped; every method receives the same clipped-linear -> standard-sRGB
input. Lens items reuse six scene/frame NPZ files without duplicating RGB.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def linear(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4).astype(np.float32)


def display(rgb):
    return np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,1e-12)**(1/2.4)-.055).astype(np.float32)


def read_exr(path):
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3 or image.shape[-1] not in [3,4]:raise ValueError('Expected RGB EXR: '+str(path))
    rgb=np.ascontiguousarray(image[...,:3][...,::-1],dtype=np.float32)
    if rgb.shape!=(512,512,3) or not np.isfinite(rgb).all():raise ValueError('Invalid full-resolution scene-linear EXR')
    return rgb


def json_write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def one_row(rows,frame,fstop=None,focus=None):
    found=[r for r in rows if int(r['frame'])==frame and (fstop is None or math.isclose(float(r['fstop']),fstop)) and (focus is None or math.isclose(float(r['focus_m']),focus))]
    if len(found)!=1:raise ValueError(f'Expected one render frame={frame}, N={fstop}, F={focus}; found{len(found)}')
    return found[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();a.record=a.record.resolve();cv2.setNumThreads(4)
    common=base/'common-inputs';manifests=base/'manifests'
    if common.exists():raise FileExistsError('Preserve prior canonical inputs: '+str(common))
    common.mkdir();fill_items=[];oracle_items=[];frames=[1,30,60];lens_rows=[];input_rows=[]
    for scene,focuses in [('B2-low',[5.3,8.4]),('B2-high',[1.4,5.])]:
        pinhole=base/'inputs'/scene/'pinhole';truth=base/'ground-truth'/scene
        pinrows=json.loads((pinhole/'timings.json').read_text());gtrows=json.loads((truth/'timings.json').read_text())
        geometry_rows=json.loads((base/scene/'geometry/summary.json').read_text())
        if len(pinrows)!=60 or sorted(int(r['frame']) for r in pinrows)!=list(range(1,61)):
            raise ValueError('All60 unique pinhole frame outputs required')
        if len(gtrows)!=18:raise ValueError('Expected3frames x3apertures x2focus complete-scene truth')
        if not all((pinhole/(r['key']+'.exr')).is_file() for r in pinrows):raise FileNotFoundError('Incomplete pinhole EXRs')
        if not all((truth/(r['key']+'.exr')).is_file() for r in gtrows):raise FileNotFoundError('Incomplete DOF EXRs')
        dest=common/scene;dest.mkdir()
        for frame in frames:
            pin=one_row(pinrows,frame);source=pinhole/(pin['key']+'.exr')
            raw=read_exr(source);clipped=np.clip(raw,0,1);rgb=np.clip(display(clipped),0,1);visible=linear(rgb)
            center_path=base/scene/f'geometry/f{frame:03d}-center.npz';bg_path=base/scene/f'geometry/f{frame:03d}-background.npz'
            center=np.load(center_path);bg=np.load(bg_path)
            raw_depth=np.asarray(center['depth_axial_m'],np.float32);known=np.asarray(center['center_visible'],bool)
            if raw_depth.shape!=(512,512) or known.shape!=raw_depth.shape or not np.isfinite(raw_depth).all() or (raw_depth<0).any():raise ValueError('Invalid first-hit depth/known map')
            if not np.array_equal(known,raw_depth>0):raise ValueError('Center hit validity and raw depth disagree')
            depth=np.where(known,raw_depth,10000.).astype(np.float32)
            alpha=np.asarray(center['foreground_alpha'],np.float32)
            if alpha.shape!=raw_depth.shape or not np.isfinite(alpha).all() or alpha.min()<0 or alpha.max()>1:raise ValueError('Invalid GT foreground coverage')
            if not np.allclose(alpha*16,np.round(alpha*16),atol=1e-5):raise ValueError('Expected4x4 GT foreground coverage')
            fill_mask=cv2.dilate((alpha>.1).astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0
            K=np.asarray(center['K'],np.float64);camera=np.asarray(center['camera_c2w'],np.float64)
            bg_depth=np.asarray(bg['depth_axial_m'],np.float32);bg_known=np.asarray(bg['center_visible'],bool)
            if bg_depth.shape!=raw_depth.shape or not np.isfinite(bg_depth).all() or (bg_depth<0).any() or not np.array_equal(bg_known,bg_depth>0):raise ValueError('Invalid background geometry')
            if not np.allclose(K,bg['K']) or not np.allclose(camera,bg['camera_c2w']):raise ValueError('Center/background camera mismatch')
            path=dest/f'f{frame:03d}.npz'
            np.savez_compressed(path,rgb_srgb=rgb,visible_rgb_linear=visible,center_depth_raw=raw_depth,
                center_surface_valid=known,center_depth=depth,GTforeground_alpha=alpha,fill_mask=fill_mask,K=K,camera_c2w=camera,
                background_depth=bg_depth,background_surface_valid=bg_known,focus_distance_m=focuses[0],
                aperture_radius_m=.05/(2*1.4),raw_rgb_exr=np.array(str(source)))
            row=dict(scene=scene,frame_id=frame,canonical_npz=str(path),raw_pinhole_exr=str(source),
                raw_linear_min=float(raw.min()),raw_linear_max=float(raw.max()),
                HDR_clipped_pixel_fraction=float(((raw<0)|(raw>1)).any(-1).mean()),
                HDR_clipped_channel_fraction=float(((raw<0)|(raw>1)).mean()),
                clipped_linear_roundtrip_max_error=float(abs(clipped-visible).max()),
                center_surface_known_pixels=int(known.sum()),method_sky_farfield_pixels=int((~known).sum()),method_sky_farfield_depth_m=10000.,
                deep_background_known_pixels=int(bg_known.sum()),deep_background_excluded_object_ids=bg['excluded_object_ids'].tolist(),
                foreground_coverage_mean=float(alpha.mean()),fill_mask_pixels=int(fill_mask.sum()),
                GT_alpha_sampling='4x4 pixel coverage, all declared static+dynamic foreground objects',
                fill_mask_rule='alpha>.1, all-ones5x5 dilation2iterations; matches official Dr.Bokeh process_bg default',
                center_geometry_source=str(center_path),background_geometry_source=str(bg_path))
            input_rows.append(row)
            item=dict(id=f'{scene}-f{frame:03d}',dataset='Blender',scene=scene,frame_id=frame,input_npz=str(path),
                uses_gt_depth=True,uses_gt_foreground_alpha=True,fill_mask_uses_gt_foreground_segmentation=True,
                fill_uses_true_hidden_depth=True,geometry_oracle_setting=True,
                unknown_sky_depth_m=10000.,unknown_sky_depth_pixels=int((~known).sum()),
                fill_mask_rule=row['fill_mask_rule'],shared_input_policy='clip raw scene-linearRGB to[0,1], standard sRGB;3D uses same inverse-derived linearRGB')
            fill_items.append(item)
            for stop in [.5,1.4,2.]:
                for focus in focuses:
                    gt=one_row(gtrows,frame,stop,focus);gt_path=truth/(gt['key']+'.exr')
                    aperture=.05/(2*stop);signed=float(K[0,0])*aperture*(1/depth-1/focus)
                    known_values=abs(signed[known]);max_radius=float(abs(signed).max())
                    entry=dict(item,id=f'{scene}-f{frame:03d}-N{stop:g}-F{focus:g}',fstop=stop,
                        focus_distance_m=focus,aperture_radius_m=aperture,gt_rgb_exr=str(gt_path),gt_samples=gt['samples'],
                        gt_seed=gt['seed'],comparison_role='pressure' if stop==.5 else 'main')
                    geometric=[v for v in geometry_rows if int(v['frame'])==frame and
                        'fstop' in v and math.isclose(float(v['fstop']),stop) and math.isclose(float(v['focus_m']),focus)]
                    if len(geometric)!=1:raise ValueError('Expected one matching optical GT geometry summary')
                    g=geometric[0];geometry_path=base/scene/'geometry'/(g['key']+'.npz')
                    if not geometry_path.exists():raise FileNotFoundError(geometry_path)
                    entry.update(geometry_gt_npz=str(geometry_path),aperture_samples=int(g['samples']),
                        low_coc_combo=bool(g['low_coc_median_below3']),
                        coc_pairmax_p10_p50_p90=g['coc_pairmax_p10_p50_p90'],
                        effective_recoverable_total_pixels=g['effective_recoverable_total_pixels'],
                        evaluation_mask_keys=dict(recoverable_total='mask_recoverable_total',
                            motion_required='mask_motion_required',parallax_sufficient='mask_parallax_sufficient',
                            unrecoverable='mask_unrecoverable',disocclusion='mask_disocclusion',residual='residual_pixels'))
                    oracle_items.append(entry)
                    lens_rows.append(dict(id=entry['id'],scene=scene,frame_id=frame,fstop=stop,focus_distance_m=focus,
                        maximum_physical_CoC_radius_all_method_pixels=max_radius,
                        maximum_physical_CoC_radius_known_surfaces=float(known_values.max()) if len(known_values) else None,
                        sky_farfield_CoC_radius=float(abs(signed[~known]).max()) if (~known).any() else None))
    maximum=max(r['maximum_physical_CoC_radius_all_method_pixels'] for r in lens_rows)
    padding=4.;kernel_offset=1.;lens=max(71,2*int(math.ceil(maximum+kernel_offset+padding))+1)
    from src.baselines.drbokeh import tail_mass
    tail=tail_mass(maximum,lens) if maximum<100 else None
    support=dict(rule='geometry-only common kernel: max(71,2*ceil(global_max_physical_radius+official_soft_offset1+tail_padding4)+1)',
        chosen_lens=lens,retained_original_lens71=lens==71,support_radius_pixels=lens//2,
        maximum_physical_radius_pixels=maximum,official_soft_kernel_radius_offset_pixels=kernel_offset,
        fixed_tail_padding_pixels=padding,numerical_discarded_radial_tail_fraction_at_global_maximum=tail,
        nominal_geometry_support_passed=maximum+kernel_offset+padding<=(lens//2),rows=lens_rows,
        selection='only GT center geometry, lens parameters and common sky convention; no method scores or predictions')
    json_write(manifests/'fill.json',dict(items=fill_items,input_policy=fill_items[0]['shared_input_policy']))
    json_write(manifests/'oracle.json',dict(items=oracle_items,drbokeh_lens=lens,input_policy=fill_items[0]['shared_input_policy'],pre_method_lens_support=support))
    record=dict(status='canonical shared inputs and method manifests prepared on CPU; methods not run',
        scenes=['B2-low','B2-high'],representative_frames=frames,pinhole_outputs_verified=120,DOF_GT_EXRs_verified=36,
        shared_frame_inputs=6,oracle_lens_items=36,LaMa_fill_items=6,inputs=input_rows,drbokeh_pre_method_support=support,
        raw_EXRs_modified=False,raw_EXR_policy='unclipped source EXR retained; never use PNG8 as evaluation target',
        evaluation_target_policy='GT EXR -> standard display sRGB/clamp[0,1] in root evaluation; no metric in this preparation script',
        frozen_criteria_created=False,method_GPU_inference=False,model_training=False,
        fill_manifest=str(manifests/'fill.json'),oracle_manifest=str(manifests/'oracle.json'))
    json_write(a.record,record);print(json.dumps(dict(shared_frame_inputs=6,oracle_items=len(oracle_items),drbokeh_lens=lens,max_CoC_radius=maximum,tail_fraction=tail)),flush=True)


if __name__=='__main__':main()
