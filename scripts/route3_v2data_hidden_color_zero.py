"""Zero-aperture checks and attribution for the hidden-colour replay."""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import torch

import route3_v2data_module2_infer as ref
from route3_v2data_E0_rerender import STAGE3C
from route3_v2data_hidden_color_sources import read


VISIBLE_FLOOR=50.
COMBINED_FLOOR=60.
VISIBLE_TOLERANCE=1e-6
START_TOLERANCE=.01
START=STAGE3C/"evaluation/step-000000/ema/report.json"
FAILURE_CASE=("development-002-9d1be93b64c29ff2","f030-focus1-N1.4","G")


@lru_cache(maxsize=1)
def start_rows():
    return {(r["scene"],r["case"]):r for r in read(START)["rows"] if r["status"]=="evaluated"}


class ZeroAperture:
    def __init__(self, visible, K, focus, center, fill, pinhole, key):
        self.visible,self.K,self.focus,self.center=visible,K,focus,center
        self.fill,self.pinhole,self.key=fill,pinhole,key
        self.full=np.ones(pinhole.shape[:2],bool)

    def score(self, hidden=None, fill=None):
        stack=ref.merge_layers(self.visible,hidden)
        rgb=ref.display(ref.raster(stack,self.K,self.focus,self.center,512,512,fill)[0])
        return ref.region_sse(rgb,self.pinhole,self.full)

    def arm(self, name, hidden=None, *, saved_combined=None):
        visible=self.score()
        complete=self.score(hidden,self.fill) if saved_combined is None else saved_combined
        if any(s["pixels"]!=self.full.size or s["psnr"] is None or not np.isfinite(s["psnr"])
               for s in (visible,complete)):
            raise ValueError(f"Invalid zero-aperture score for {self.key}/{name}")
        below=complete["psnr"]<COMBINED_FLOOR
        if name=="Query" and below:
            raise ValueError(f"Original Query zero-aperture floor failed for {self.key}: {complete['psnr']}")
        return dict(visible_zero=visible,zero=complete,below_60_db=below)

    def changed_rays(self, original, removed, recolored):
        # A pinhole sample contributes iff its marker survives front-to-back
        # alpha compositing on the original, unfiltered scene.
        marks=torch.stack((removed,recolored),dim=-1).float()
        front=replace(self.visible,colors=self.visible.colors.new_zeros((len(self.visible.means),2)))
        scene=ref.merge_layers(front,replace(original,colors=marks))
        weights=ref.raster(scene,self.K,self.focus,self.center,512,512)[0]
        def count(tolerance):
            flags=weights>tolerance
            return dict(removed=int(flags[...,0].sum()),recolored=int(flags[...,1].sum()),
                        either=int(flags.any(-1).sum()))
        return dict(total_rays=self.full.size,positive_compositing_weight=count(0),
                    weight_above_1e_6=count(VISIBLE_TOLERANCE))


def verify_case(row, *, cached_visible=None):
    """Check the entire same-geometry group across methods and focus protocols."""
    geometry_scores={"V0_pred":[],"V0_GT":[],"E2k":[]}
    for convention, group in row["render"].items():
        for family, arms in group.items():
            geometry="V0_pred" if family=="Query" else family
            arms=arms["methods"] if family=="Query" else arms
            geometry_scores[geometry].extend(r["visible_zero"]["psnr"] for r in arms.values())
    if cached_visible:
        geometry_scores["V0_pred"].extend(cached_visible)
    label=(row["scene"],row["case"])
    spreads={geometry:max(scores)-min(scores) for geometry,scores in geometry_scores.items()}
    for geometry, scores in geometry_scores.items():
        if min(scores)<VISIBLE_FLOOR or spreads[geometry]>=VISIBLE_TOLERANCE:
            raise ValueError(f"Visible zero-aperture check failed for {label}/{geometry}: "
                             f"minimum {min(scores)}, spread {spreads[geometry]}")
    original=row["render"]["P"]["Query"]["methods"]["Query"]["zero"]
    archived=start_rows()[label]
    if abs(row["focus_error"]["V0_pred"]["focus_m"]-archived["focus_m"])>VISIBLE_TOLERANCE:
        raise ValueError(f"Archived 3c focus does not match the P convention for {label}")
    delta=abs(original["psnr"]-archived["zero_combined"]["psnr"])
    if delta>START_TOLERANCE:
        raise ValueError(f"Original Query P zero-aperture differs from 3c for {label}: {delta}")
    return dict(visible_max_delta_db=spreads,query_P_3c_abs_delta_db=delta,
                archived_query_P_db=archived["zero_combined"]["psnr"])
