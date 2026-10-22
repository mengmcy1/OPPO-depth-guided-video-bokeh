"""CPU geometry label fitting, controlled perturbation and paired data access."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def bounded_affine(x, y):
    """Exact 2-variable least-squares on 0 <= low <= high <= 1.

    All support pixels enter the moments. x is inverse depth, y fixed VDA.
    Enumerate unconstrained solution and the three edges of the triangle.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    xmin, xmax = float(x.min()), float(x.max())
    assert xmax > xmin and x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    t = (x-xmin)/(xmax-xmin)
    u = 1-t
    h = np.array([[np.mean(u*u), np.mean(u*t)], [np.mean(u*t), np.mean(t*t)]])
    q = np.array([np.mean(u*y), np.mean(t*y)])
    assert np.linalg.det(h) > 1e-12, 'Degenerate inverse depth support'
    free = np.linalg.solve(h, q)
    assert free[1] > free[0], 'Nonpositive GT/VDA association; cannot label automatically'
    candidates = [np.array([0., np.clip(q[1]/h[1,1], 0, 1)]),
                  np.array([np.clip((q[0]-h[0,1])/h[0,0], 0, 1), 1.]),
                  np.full(2, np.clip(y.mean(), 0, 1))]
    if 0 <= free[0] <= free[1] <= 1:
        candidates.append(free)
    best = min(candidates, key=lambda v: float(v@h@v - 2*q@v))
    assert best[1] > best[0] + 1e-8, 'Constrained solution collapsed; needs review'
    def params(v):
        a = float((v[1]-v[0])/(xmax-xmin))
        return dict(scale=a, shift=float(v[0]-a*xmin), endpoint_low=float(v[0]),
                    endpoint_high=float(v[1]), inverse_min=xmin, inverse_max=xmax)
    return params(free), params(best)


def map_reference(inv, domain, params):
    # Endpoint form avoids a cancellation-generated negative value at the lower bound.
    t = (inv.astype(np.float64)-params['inverse_min'])/(params['inverse_max']-params['inverse_min'])
    mapped = (params['endpoint_low'] + t*(params['endpoint_high']-params['endpoint_low'])).astype(np.float32)
    mapped[~domain] = 0  # Invalid placeholder, never a supervised zero-depth label.
    assert np.isfinite(mapped).all()
    return mapped


def rng_for(seed, sequence, frame, purpose):
    digest = hashlib.sha256(f'{seed}:{sequence}:{frame}:{purpose}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def frame_signs(seed, sequence, frames):
    assert len(frames) == 25
    signs = np.array([0]*5 + [-1]*10 + [1]*10, dtype=np.int8)
    rng_for(seed, sequence, 0, 'signs').shuffle(signs)
    return dict(zip(frames, map(int, signs)))


def perturb(d, domain, sequence, frame, sign, config):
    assert d.dtype == np.float32 and domain.dtype == bool and d.shape == domain.shape
    h, w = d.shape
    mask = np.zeros((h,w), bool)
    weight = np.zeros((h,w), np.float32)
    info = dict(requested_sign=int(sign), status='zero_requested', amplitude=0., attempts=0)
    if sign == 0:
        return d.copy(), mask, weight, info
    rng = rng_for(config['seed'], sequence, frame, 'ellipse')
    amplitude = float(rng.uniform(*config['amplitude_range']))
    for attempt in range(config['max_attempts']):
        area = float(rng.uniform(*config['area_fraction_range']))
        ratio = float(rng.uniform(*config['axis_ratio_range']))
        rx = np.sqrt(area*h*w*ratio/np.pi)
        ry = np.sqrt(area*h*w/(ratio*np.pi))
        if 2*rx+2 >= w or 2*ry+2 >= h:
            continue
        cx, cy = float(rng.uniform(rx+1,w-rx-1)), float(rng.uniform(ry+1,h-ry-1))
        yy, xx = np.ogrid[:h,:w]
        r = np.sqrt(((xx-cx)/rx)**2 + ((yy-cy)/ry)**2)
        candidate = r < 1
        if not candidate.any() or not domain[candidate].all():
            continue
        mask = candidate
        ramp = np.clip((r-config['flat_radius'])/(1-config['flat_radius']), 0, 1)
        weight = (0.5*(1+np.cos(np.pi*ramp))).astype(np.float32)
        weight[~mask] = 0
        info.update(status='applied', amplitude=amplitude, attempts=attempt+1,
                    center_xy=[cx,cy], radii_xy=[rx,ry], requested_area_fraction=area)
        break
    else:
        info.update(status='no_valid_ellipse', attempts=config['max_attempts'])
    raw = d + np.float32(sign*amplitude)*weight
    out = np.clip(raw, 0, 1).astype(np.float32)
    info.update(clipped_pixels=int(((raw<0)|(raw>1)).sum()),
                actual_changed_pixels=int((out!=d).sum()),
                actual_max_delta=float(np.abs(out-d).max()))
    assert np.array_equal(out[~mask], d[~mask]) and not mask[~domain].any()
    return out, mask, weight, info


def add_residual(actual_input, residual):
    """Pre-clamp output; the caller must supply the actual natural/perturbed input."""
    assert actual_input.shape == residual.shape
    return actual_input + residual


class GeometryPairs:
    """Small NumPy/PIL reader. No model import and no GPU or GT in input channels."""
    def __init__(self, repository, manifest, split):
        self.root = Path(repository)
        self.manifest = json.loads((self.root/manifest).read_text())
        assert split in ('pilot_train','pilot_dev')
        self.entries = [r for r in self.manifest['frames'] if r['split']==split]
        self._name = None

    def __len__(self):
        return len(self.entries)

    def pair(self, index):
        row = self.entries[index]
        if self._name != row['sequence']:
            with np.load(self.root/row['vda_npz']) as z:
                self._natural = z['disparity_normalized']
            with np.load(self.root/row['pair_npz']) as z:
                self._paired = {k:z[k] for k in z.files}
            self._name = row['sequence']
        i = row['array_index']
        with Image.open(self.root/row['rgb']) as im:
            rgb = np.asarray(im, dtype=np.float32).transpose(2,0,1)/255.
        natural = self._natural[i]
        altered = self._paired['perturbed'][i]
        target = self._paired['reference'][i]
        valid = self._paired['supervision_domain'][i]
        normal = self._paired['near_reference'][i]
        mask = self._paired['perturbation_mask'][i]
        # Identical supervision object for the two branches. Keep constraint is only
        # applied to untouched near-reference pixels on the perturbed branch.
        a = dict(input=np.concatenate([rgb,natural[None]],0), target=target,
                 valid=valid, keep=normal)
        b = dict(input=np.concatenate([rgb,altered[None]],0), target=target,
                 valid=valid, keep=normal&~mask)
        return a,b,row
