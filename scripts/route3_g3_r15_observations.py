"""World-point scoring for actual packed observations, separate from inference."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from route3_s2a_proxy import read as read_exr


def project(camera, K):
    matrix = camera @ K.T
    return matrix[..., :2] / np.maximum(matrix[..., 2:], 1e-8)


def pixels(uv):
    integer = np.floor(uv).astype(np.int32)
    inside = ((integer >= 0) & (integer < 512)).all(-1)
    return integer.clip(0, 511), inside


def surfaces(root, frames):
    observations = []
    for f in frames:
        path = root / 'input-passes-v2'
        world = read_exr(path / f'f{f:03d}-Position-{f:04d}.exr')
        obj = read_exr(path / f'f{f:03d}-IndexOB-{f:04d}.exr')[..., 0].round().astype(np.int16)
        footprint = np.full(obj.shape, 1e-4, np.float32)
        footprint[:-1] = np.maximum(footprint[:-1], np.where(obj[:-1] == obj[1:],
                                    np.linalg.norm(world[:-1] - world[1:], axis=-1), 0))
        footprint[:, :-1] = np.maximum(footprint[:, :-1], np.where(obj[:, :-1] == obj[:, 1:],
                                        np.linalg.norm(world[:, :-1] - world[:, 1:], axis=-1), 0))
        observations.append((world, obj, footprint))
    return observations


@dataclass
class Observations:
    world: np.ndarray
    obj: np.ndarray
    camera: np.ndarray
    center_hidden: np.ndarray
    center_xy: np.ndarray
    in_center: np.ndarray
    radius: np.ndarray
    slots: np.ndarray
    xy: np.ndarray
    context_pose: np.ndarray
    target_pose: np.ndarray
    K: np.ndarray

    @classmethod
    def from_sources(cls, native, context_pose, target_slot, K, depth, slots, xy):
        world = np.empty((len(slots), 3), np.float32)
        obj = np.empty(len(slots), np.int16)
        radius = np.empty(len(slots), np.float32)
        for i, (pos, ids, footprint) in enumerate(native):
            use = slots == i
            x, y = xy[use].T
            world[use], obj[use], radius[use] = pos[y, x], ids[y, x], footprint[y, x]
        pose = context_pose[target_slot]
        cam = (world - pose[:3, 3]) @ pose[:3, :3]
        ij, inside = pixels(project(cam, K))
        x, y = ij.T
        inside &= cam[:, 2] > 0
        center_distance = np.linalg.norm(world - native[target_slot][0][y, x], axis=-1)
        hidden = inside & (obj > 0) & (depth[y, x] > 0)
        hidden &= (cam[:, 2] > depth[y, x] + 1e-4) & (center_distance > 1.5 * radius + 1e-4)
        return cls(world, obj, cam, hidden, ij, inside, radius, slots, xy, context_pose, pose, K)

    def hit_fraction(self, geo, nominal_focus):
        K, pose = self.K, self.target_pose
        if 'world_position' in geo:
            mesh_world = geo['world_position']
        else:
            uv = geo['reference_uv_center']
            rays = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], -1) @ np.linalg.inv(K).T
            camera = rays * geo['depth_axial_m'][..., None]
            mesh_world = (camera @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
        counter = np.zeros(len(self.slots), np.uint8)
        hidden, ids = geo['hidden'], geo['object_id']
        for k, xy in enumerate(geo['aperture_xy']):
            camera = self.camera - np.r_[xy, 0.]
            intr = K.copy()
            intr[:2, 2] += np.diag(K)[:2] * xy / nominal_focus
            ij, inside = pixels(project(camera, intr))
            x, y = ij.T
            hit = mesh_world[k, y, x]
            good = inside & self.center_hidden & (camera[:, 2] > 0) & hidden[k, y, x]
            good &= (ids[k, y, x] == self.obj) & (np.linalg.norm(self.world - hit, axis=-1) <= 1.5 * self.radius + 1e-4)
            for slot in np.unique(self.slots[good]):
                at = np.flatnonzero(good & (self.slots == slot))
                p = self.context_pose[slot]
                source_cam = (hit[at] - p[:3, 3]) @ p[:3, :3]
                good[at] &= (abs(project(source_cam, K) - (self.xy[at] + .5)) <= .75).all(-1)
            counter += good
        return counter.astype(np.float32) / 16

    def placement(self, fit, depth, rule):
        from scipy.ndimage import maximum_filter

        x, y = self.center_xy.T
        inverse = fit['a'][y, x] / np.maximum(self.camera[:, 2], 1e-6) + fit['b'][y, x]
        valid = self.in_center & fit['local_supported'][y, x] & np.isfinite(inverse) & (inverse > 1e-4)
        mapped = 1 / np.maximum(inverse, 1e-4)
        bound = 1.03 * depth[y, x]
        violation = mapped < bound
        if rule == 'O1':
            mapped = np.maximum(mapped, bound)
        else:
            bg = maximum_filter(depth, 49)[y, x]
            valid &= ~violation | (bg >= bound)
            mapped = np.where(violation, bg, mapped)
        camera = self.camera * (mapped / np.maximum(self.camera[:, 2], 1e-6))[:, None]
        return mapped.astype(np.float32), camera.astype(np.float32), valid

    def colors(self, parent):
        result = np.zeros_like(self.world)
        supported = np.zeros(len(result), bool)
        radius = 3 * np.maximum(self.camera[:, 2], 0) / self.K[0, 0] + 1e-4
        for oid in np.unique(self.obj[self.obj > 0]):
            points = np.flatnonzero(self.obj == oid)
            pool = np.flatnonzero((parent['object_id'] == oid) & parent['recoverable'])
            if not len(pool):
                continue
            distance, index = cKDTree(parent['world_position'][pool]).query(self.world[points], workers=2)
            supported[points] = distance <= radius[points]
            result[points] = parent['colors_linear'][pool[index]]
        return result, supported
