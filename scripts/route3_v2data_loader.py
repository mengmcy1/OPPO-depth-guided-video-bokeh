"""Training split adapters for the expanded route 3 data."""
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
import hashlib
import math
import re

import numpy as np
import torch

import route3_ovg_data as legacy
from route3_ovg_1c import aligned_sample
from route3_v2data_a_build import OnlineA
from route3_v2data_B_inputs import FORMAL_DATA, library_unit_paths
from route3_v2data_G_priors import inventory as g_inventory, sample as g_sample
from route3_v2data_R_priors import inventory as r_inventory, training_input as r_sample
from route3_v2data_c_build import OUT as C_ROOT, optical_specs as c_optical_specs, window_inventory
from src.ovg_edge_statistics import adjacent_surface_pairs

ROOT = legacy.ROOT
OPTICAL_FRAMES = (1, 30, 60)
FSTOPS = (1.4, 2.0)
MIX = {
    '1a': {'G': 60, 'C': 25, 'B': 15},
    '1b': {'C': 40, 'B': 35, 'G': 15, 'R': 10},
}


@lru_cache(maxsize=16)
def read(path):
    return legacy.read(Path(path))


def registered_b():
    """Use only released training geometry and omit unrepaired curved units."""
    top = legacy.read(FORMAL_DATA/'manifest.json')
    registered = {r['scene_id'] for r in top['units'] if r['partition'] == 'training'}
    allowed, excluded = {}, {}
    for unit in library_unit_paths('train'):
        meta = legacy.read(unit/'manifest.json')
        if meta['partition'] != 'training':
            raise ValueError(f'Unexpected partition for {unit.name}')
        if unit.name not in registered:
            excluded[unit.name] = 'awaiting release registration'
            continue
        patch = re.search(r'-p([0-9]+)$', meta['generator_version'])
        if meta['category'] == 'curved_self_occlusion' and (not patch or int(patch[1]) < 2):
            excluded[unit.name] = 'curved self-occlusion awaits p2'
            continue
        if not (unit/'observations/priors/summary.json').is_file():
            excluded[unit.name] = 'observation priors pending'
            continue
        allowed[unit.name] = unit
    return allowed, excluded


def schedule(stage, rng, minimum=20):
    """Exact seeded mixture. All scheduled positions survive partial releases."""
    ratios = {tier: Fraction(percent, 100) for tier, percent in MIX[stage].items()}
    block = math.lcm(*(r.denominator for r in ratios.values()))
    length = math.ceil(max(minimum/r for r in ratios.values())/block)*block
    sequence = [tier for tier, ratio in ratios.items() for _ in range(int(length*ratio))]
    return rng.permutation(sequence).tolist()


class TrainingData:
    def __init__(self):
        self.a = OnlineA('train')
        self.b, self.b_excluded = registered_b()
        self.items = {
            'A': [r['key'] for r in self.a.foreground],
            'B': [('B', u, i) for u in self.b for i in range(len(legacy.TRAIN_FRAMES))],
            'C': window_inventory('train'),
            'G': list(g_inventory('train')),
            'R': list(r_inventory('train')),
        }
        self.allowed = {tier: set(items) for tier, items in self.items.items() if tier != "A"}
        groups = {}
        for item in self.items['G']:
            environment = read(item[1])['environment']
            groups.setdefault(environment, []).append(item)
        self.g_groups = [groups[k] for k in sorted(groups)]

    def choose(self, tier, rng):
        if not self.items[tier]:
            return None
        rows = self.items[tier]
        if tier == 'G':
            rows = self.g_groups[rng.integers(len(self.g_groups))]
        return rows[rng.integers(len(rows))]

    def load(self, item, device):
        tier, key, index = item
        if item not in self.allowed.get(tier, ()):
            raise ValueError(f'Sample is outside the eligible training inventory: {item}')
        if tier == 'B':
            return aligned_sample(item, device)
        if tier == 'G':
            return g_sample(item, device)
        if tier == 'R':
            return r_sample(key, index, device)
        if tier != 'C':
            raise ValueError(tier)
        path = Path(key)
        if path.parent.parent != C_ROOT/'windows/train' or read(path)['split'] != 'train':
            raise ValueError(f'C sample is outside the training pool: {path}')
        x, y = legacy.sample(item, device)
        return x, {k: v.float() if v.is_floating_point() else v for k, v in y.items()}

    def optical_specs(self, item):
        tier, key, index = item
        if tier == 'C':
            specs = c_optical_specs(key, index)
            if specs and (len(specs) != 2 or any(not s['truth'].is_file() or not s['label'].is_file()
                                                 for s in specs)):
                raise ValueError(f'Incomplete C optical references for {key}')
            return specs
        if tier != 'B' or key not in self.b:
            return []
        frame = legacy.TRAIN_FRAMES[index]
        if frame not in OPTICAL_FRAMES:
            return []
        return [row for row in b_optical_cases(self.b[key]) if row['frame'] == frame]

    def pair_reference(self, item, target):
        tier, key, index = item
        if tier == 'B':
            return legacy.b_edge_reference(item, target)
        meta = read(key)
        recipe = read(ROOT/meta['geometry_label_recipe'])
        fields = legacy.load(ROOT/meta['rows'][index]['label'], ('center_surface_id',))
        ids = fields['center_surface_id']
        origin = np.asarray(meta['rows'][index]['c2w'], np.float32)[2, 3]
        geometry_depth = np.asarray(recipe['rectangles'], np.float32)[:,0] - origin
        valid = target['valid'][0].cpu().numpy() & (ids > 0)
        return dict(gt=geometry_depth[np.maximum(ids, 1)-1], object_id=ids, valid=valid,
                    foreground=np.arange(1, recipe['front_layers']+1),
                    boundary=target['depth_boundary'][0].cpu().numpy())


def b_optical_cases(unit):
    """Resolve only a complete, geometry-matched training optical release."""
    optical = FORMAL_DATA/'optical'/unit.name
    manifest = optical/'manifest.json'
    if not manifest.is_file():
        return []
    release = legacy.read(manifest)
    digest = hashlib.sha256((unit/'manifest.json').read_bytes()).hexdigest()
    if release.get('partition') != 'training' or release.get('geometry_manifest_sha256') != digest:
        raise ValueError(f'Stale or non-training optical release: {unit.name}')
    labels = optical/'labels'
    rows = legacy.read(labels/'optical-audit.json')['optical']
    expected = {f'f{frame:03d}-focus{focus}-N{n:g}' for frame in OPTICAL_FRAMES
                for focus in (0, 1) for n in FSTOPS}
    if set(r['id'] for r in rows) != expected or len(rows) != len(expected):
        raise ValueError(f'Incomplete B aperture/focus grid: {unit.name}')
    specs = []
    for row in rows:
        frame_tag, focus_tag, _ = row['id'].split('-')
        frame = int(frame_tag[1:])
        truth = labels/f"{row['id']}-optics.npz"
        if not truth.is_file():
            raise FileNotFoundError(truth)
        # Native f/1.4 K16 is stored with the geometry release. A different pupil
        # needs its own traced rays, which the optics adapter computes in memory.
        rays = unit/'labels'/f'{frame_tag}-{focus_tag}-K16.npz'
        native = rays if row['fstop'] == 1.4 and rays.is_file() else None
        specs.append(dict(tiers='B', frame=frame, focus=row['focus_m'], fstop=row['fstop'],
                          truth=truth, label=native, unit=unit))
    return sorted(specs, key=lambda row: (row['frame'], row['focus'], row['fstop']))


def same_surface_loss(prediction, reference):
    """L1 residual of predicted and GT native inverse-depth differences."""
    inv = prediction['inverse1'][0]
    gt = reference['gt'].astype(np.float64)
    inv_gt = np.divide(1., gt, out=np.zeros_like(gt), where=gt>0)
    total = inv.sum()*0
    counts = {}
    for direction, p, q, continuous, crossing in adjacent_surface_pairs(**reference):
        mask = torch.as_tensor(continuous, device=inv.device)
        delta = inv[q]-inv[p] - inv.new_tensor(inv_gt[q]-inv_gt[p])
        total = total + delta[mask].abs().sum()
        counts[direction] = int(continuous.sum())
        counts[f'excluded_crossing_{direction}'] = int(crossing.sum())
    return total/max(1, counts['right']+counts['down']), counts
