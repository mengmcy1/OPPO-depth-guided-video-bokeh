"""Build fixed-camera C1 proxy masks, without loading any MoVieS output."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camvfx4d import read_rgb, load_camera_frames, intrinsics_from_fov, effect_parameters
from src.c1_regions import background_and_motion, proxy_regions, mask_summary, linear_rgb, display_rgb
from route3_mesh_probe import cast


def geometry_probe(ray, camera, K, size, parameters, config):
    y, x = np.mgrid[:size, :size]
    rays = np.stack([x.ravel() + .5, y.ravel() + .5, np.ones(x.size)], -1) @ np.linalg.inv(K).T
    center, valid = cast(ray, camera[:3, 3], rays @ camera[:3, :3].T)
    depth = ((center - camera[:3, 3]) @ camera[:3, :3])[:, 2]
    positive = valid & (depth > 1e-4)
    clean_depth = np.where(positive, depth, 0).reshape(size, size).astype(np.float32)
    nearest, farthest = float(depth[positive].min()), float(depth[positive].max())
    focus, aperture = parameters['focus_distance_m'], parameters['aperture_radius_m']
    # This is a radius, not a diameter. Include the far endpoint, since focus can be near.
    radius_float = K[0, 0] * aperture * max(abs(1 / nearest - 1 / focus), abs(1 / farthest - 1 / focus))
    radius = max(1, int(math.ceil(radius_float)))
    count = config['aperture_samples']; half = count // 2; index = np.arange(half)
    r = np.sqrt((index + .5) / half); theta = index * math.pi * (3 - math.sqrt(5))
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], -1)
    xy = np.concatenate([xy, -xy]) * aperture
    any_occluded = np.zeros(x.size, bool); known = np.zeros(x.size, np.uint8)
    for offset2 in xy:
        offset = np.r_[offset2, 0.]
        hits, good = cast(ray, camera[:3, 3] + camera[:3, :3] @ offset,
                          (rays * focus - offset) @ camera[:3, :3].T)
        cp = (hits - camera[:3, 3]) @ camera[:3, :3]
        projected = cp @ K.T
        uv = projected[:, :2] / projected[:, 2:]
        inside = good & (cp[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
        ids = np.flatnonzero(inside)
        first, ok = cast(ray, camera[:3, 3], hits[ids] - camera[:3, 3])
        distance = np.linalg.norm(hits[ids] - camera[:3, 3], axis=-1)
        first_distance = np.linalg.norm(first - camera[:3, 3], axis=-1)
        hidden = ok & (distance - first_distance > np.maximum(config['mesh_visibility_absolute_tolerance_m'],
                                                               distance * config['mesh_visibility_relative_tolerance']))
        known[ids[ok]] += 1; any_occluded[ids[hidden]] = True
    return clean_depth, any_occluded.reshape(size, size), known.reshape(size, size) / count, dict(
        nearest_mesh_axial_depth_m=nearest, farthest_mesh_axial_depth_m=farthest,
        maximum_coc_radius_pixels=float(radius_float), conservative_disk_radius_pixels=radius,
        center_mesh_coverage=float(positive.mean()), mesh_time='unconfirmed; static geometry proxy')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/raw/CamVFX-4D'))
    p.add_argument('--config', type=Path, default=Path('configs/route3-c1-a1-20261110.json'))
    p.add_argument('--scenes', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); c = json.loads(a.config.read_text()); cv2.setNumThreads(4)
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(config=c, method_outputs_used=False, scenes=[],
                  limitations=['median agreement is not hidden-background truth',
                               'shadows/reflections can enter motion masks',
                               'depth proxy is one unconfirmed-time mesh',
                               'motion maximum CoC is a conservative disk; not aperture visibility GT'])
    size = c['resolution']; chosen = c['input_frame_ids']; selected = np.array(chosen) - 1
    for scene_name in a.scenes:
        started = time.monotonic(); scene = a.raw / scene_name
        frames, meta = load_camera_frames(scene / 'pinhole/transforms_train.json')
        K = intrinsics_from_fov(meta, size, size); parameters = effect_parameters(scene)[0]
        mesh = trimesh.load(scene / 'pinhole/points3d_opencv.ply', process=False)
        mesh.vertices = np.asarray(mesh.vertices) * np.asarray(c['mesh_axis_transform'])
        ray = RayMeshIntersector(mesh); rows = []; views = []
        for view in c['views']:
            out = a.output / scene_name / str(view); out.mkdir(parents=True)
            images = np.stack([linear_rgb(read_rgb(scene / f'pinhole/{view}/{i:06d}.png')) for i in range(1, 51)])
            board, agreement, valid, motion, raw_motion = background_and_motion(images, c)
            depth, static_disocclusion, known, geometry = geometry_probe(ray, frames[(view, 1)]['c2w'], K, size, parameters, c)
            cv2.imwrite(str(out / 'background-board.png'), np.uint8(display_rgb(board)[..., ::-1] * 255))
            np.savez_compressed(out / 'background.npz', linear_rgb=board, agreement_frames=agreement,
                                valid=valid, motion_masks=motion, raw_motion_masks=raw_motion,
                                mesh_depth=depth, static_disocclusion=static_disocclusion, known_ray_fraction=known, K=K)
            frames_summary = []; all_regions = []
            for frame_id in chosen:
                reg = proxy_regions(motion, valid, frame_id - 1, selected, geometry['conservative_disk_radius_pixels'],
                                    static_disocclusion, known >= .5, c)
                all_regions.append(reg)
                stats = {name: mask_summary(reg[name], reg['confidence'] if name != 'static_static' else known)
                         for name in ['M_t', 'D_t', 'M_C1_proxy', 'dynamic_unrecoverable', 'dynamic_unknown', 'static_static', 'M_C1_proxy_50']}
                stats['frame_id'] = frame_id
                stats['rho13_mean_in_D'] = float(reg['rho_t'][reg['D_t']].mean()) if reg['D_t'].any() else None
                stats['rho50_mean_in_D'] = float(reg['rho_50'][reg['D_t']].mean()) if reg['D_t'].any() else None
                stats['additional_recoverable_pixels_with_50'] = int((reg['M_C1_proxy_50'] & ~reg['M_C1_proxy']).sum())
                frames_summary.append(stats); rows.append(stats)
            np.savez_compressed(out / 'regions.npz', frame_ids=chosen,
                                **{name: np.stack([r[name] for r in all_regions]) for name in all_regions[0]})
            montage = []
            for j in [0, 6, 12]:
                reg = all_regions[j]; rgb = display_rgb(images[chosen[j] - 1]).copy()
                overlay = rgb.copy()
                overlay[reg['M_C1_proxy']] = [0, 1, 0]
                overlay[reg['dynamic_unrecoverable']] = [1, 0, 0]
                overlay[reg['dynamic_unknown']] = [1, 0, 1]
                overlay[reg['static_static']] = [0, .5, 1]
                montage.append(np.concatenate([rgb, .5 * rgb + .5 * overlay], 1))
            cv2.imwrite(str(out / 'regions-preview.png'), np.uint8(np.concatenate(montage, 0)[..., ::-1] * 255))
            diagnostics = dict(view=view, geometry=geometry, background_valid_fraction=float(valid.mean()),
                agreement_frames_quantiles=np.quantile(agreement, [0, .1, .5, .9, 1]).tolist(),
                ever_changed_fraction=float(motion.any(0).mean()),
                majority_motion_fraction=float((motion.sum(0) > 25).mean()),
                no_motion_observed_fraction=float((~motion.any(0)).mean()),
                note='photometric consistency cannot distinguish persistent foreground from background', frames=frames_summary)
            views.append(diagnostics)
            print(json.dumps(dict(scene=scene_name, view=view, ever_changed=diagnostics['ever_changed_fraction'],
                                  background_valid=diagnostics['background_valid_fraction'],
                                  c1_pixels=sum(r['M_C1_proxy']['valid_pixels'] for r in frames_summary), geometry=geometry)), flush=True)
            del images
        summary = dict(scene=scene_name, split='development' if scene_name in c['development_scenes'] else 'test_mask_only',
                       views=views, elapsed_seconds=time.monotonic() - started, regions={})
        for name in ['M_C1_proxy', 'dynamic_unrecoverable', 'static_static']:
            pixels = sum(r[name]['valid_pixels'] for r in rows)
            times = sum(any(v['frames'][i][name]['valid_pixels'] > 0 for v in views) for i in range(len(chosen)))
            summary['regions'][name] = dict(total_pixels=pixels, distinct_effective_times=times,
                fraction_over_all_evaluated_pixels=pixels / (len(c['views']) * len(chosen) * size * size),
                meets_candidate_minimum=pixels >= c['minimum_scene_pixels_candidate'] and times >= c['minimum_scene_times_candidate'])
        result['scenes'].append(summary)
        (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
