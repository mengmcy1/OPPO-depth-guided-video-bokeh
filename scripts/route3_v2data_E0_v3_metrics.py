"""Scheme A reachability, common holdout selection and component diagnostics."""
from collections import Counter, defaultdict
import math

import numpy as np
import torch

import route3_v2data_E0_train as training
from route3_v2data_E0_data import auxiliary_path, mapped
from route3_v2data_E_labels import stats
from src.ovg_e import ORDER_RATIO, RESIDUAL_LIMIT


COC_K=25.4
SEE_SOURCE="https://github.com/zhyever/PatchFusion/blob/main/estimator/utils/metric.py"
RULE=dict(metric="Equal mean of the rod, gap, boundary and normal zone pixel means of 25.4 * abs(decoded rho - GT center rho), in px.",
          boundary="For a boundary prediction at p, take the minimum error against valid ground truth at q in the 3x3 window about p.",
          see_source=SEE_SOURCE,rod_gate="Holdout rod absolute decoded-error p50 <= 1.00 * OVG rho1 rod-error p50.",
          eligible_tie="Earlier step",fallback="Minimum rod p50, then earlier step if tied.",
          diagnostic_only=["NLL","spatial_switch"],checkpoints="EMA every 1000 steps from 1000 to 20000")


def valid_labels(visible,dual,zone,center,back):
    if (not np.isfinite(center[visible]).all() or (center[visible]<=0).any()
            or np.any(zone[visible]>3) or np.any(zone[visible]<0)
            or not np.isfinite(back[dual]).all() or (back[dual]<=0).any()):
        raise ValueError("Invalid visible/dual inverse depth or loss-zone labels")


def reachability(rows, gate):
    counts=defaultdict(Counter)
    for row in rows:
        path=auxiliary_path(row["scene"],row["frame"])
        visible,dual,center,back,zone,base,rho,attenuation,source=(
            mapped(path,k)[0] for k in ("visible","valid_dual","center","back","loss_zone","beta",
                                       "rho1","attenuation","mv_source"))
        valid_labels(visible,dual,zone,center,back)
        single=visible&~dual
        if ((source>2)&single).any() or not all(np.isfinite(v[single]).all() for v in (base,rho,attenuation)):
            raise ValueError(f"Invalid rear anchors for {row['scene']}/{row['frame']}")
        # Both tanh residual biases are zero at initialization. Values at the
        # order ceiling are reachable whenever the truncated interval includes it.
        ceiling=np.log(np.maximum(rho,1e-8))+math.log(ORDER_RATIO)
        target=np.minimum(np.log(np.maximum(center,1e-8)),ceiling)
        ranges={"v2_initial":RESIDUAL_LIMIT*attenuation*gate,
                "v3_limit":RESIDUAL_LIMIT*np.where(source==0,1.,attenuation)}
        for label,active in (("rho2",single&(source==0)),("multiview",single&(source!=0))):
            result=counts[label]
            result["pixels"]+=int(active.sum())
            for name,limit in ranges.items():
                lo=np.minimum(base-limit,ceiling)
                hi=np.minimum(base+limit,ceiling)
                result[name]+=int((active&((target<lo)|(target>hi))).sum())
    return dict(frames=len(rows),initial_gate=float(gate),
                assumptions="Initialize mu_f at log rho1; evaluate the post-order-clamp interval. V2 uses initial g, V3 multiview uses g=1 as the attainable upper limit.",
                by_source={k:dict(n=v["pixels"],v2_initial_unreachable=v["v2_initial"],v3_unreachable=v["v3_limit"],
                    v2_initial_fraction=training.fraction(v["v2_initial"],v["pixels"]),
                    v3_fraction=training.fraction(v["v3_limit"],v["pixels"])) for k,v in counts.items()})


def soft_boundary_error(pred, gt, valid):
    h,w=gt.shape
    padded=np.pad(np.where(valid,gt,np.inf),1,constant_values=np.inf)
    error=np.full_like(gt,np.inf)
    for y in range(3):
        for x in range(3):
            np.minimum(error,np.abs(pred-padded[y:y+h,x:x+w]),out=error)
    return error


class Diagnostics:
    def __init__(self):
        self.baseline=training.new_diagnostics()
        self.values=defaultdict(list)
        self.saturation=defaultdict(Counter)
        self.selection=defaultdict(Counter)
        self.switch=Counter()

    def add(self, out, target):
        def plane(tensor):
            return tensor[0,0].detach().cpu().numpy()
        visible,dual,zone,center,back,source=(plane(target[k]) for k in
            ("visible","valid_dual","loss_zone","center","back","mv_source"))
        valid_labels(visible,dual,zone,center,back)
        training.add_diagnostics(self.baseline,out,target)
        pred=plane(out["decoded"])
        picks=plane(out["selected_back"])
        raw_back=plane(out["raw_back"])
        unambiguous=np.abs(pred-center)
        soft=soft_boundary_error(pred,center,visible)
        saturation=np.abs(plane(out["raw_tanh"][:,1:2]))>.95
        back_target=plane(training.single_back_target(out,target))
        for axis in (0,1):
            support=np.swapaxes(visible,0,axis)
            chosen=np.swapaxes(picks,0,axis)
            pair=support[1:]&support[:-1]
            self.switch.update(pairs=int(pair.sum()),changed=int((pair&(chosen[1:]!=chosen[:-1])).sum()))
        for i,group in enumerate(training.GROUPS):
            mask=visible&(zone==i)
            coc=soft if group=="boundary" else unambiguous
            self.selection[group].update(n=int(mask.sum()),sum=float(coc[mask].sum(dtype=np.float64)*COC_K))
            for layer,roi,labels in (("single",mask&~dual,back_target),
                                      ("dual",mask&dual,np.log(np.maximum(back,1e-8)))):
                self.values[f"{group}/rear.{layer}"].append((raw_back-labels)[roi])
            self.values[f"{group}/front_scale"].append(plane(out["scales"][:,:1])[mask])
            self.values[f"{group}/back_scale"].append(plane(out["scales"][:,1:2])[mask])
            for label,places in (("rho2",source==0),("multiview",source!=0),
                                  ("DS",source==1),("tri",source==2)):
                region=mask&places
                self.saturation[f"{group}/{label}"].update(n=int(region.sum()),
                                                          saturated=int((region&saturation).sum()))

    def summary(self):
        scores={k:dict(n=v["n"],mean_px=training.fraction(v["sum"],v["n"])) for k,v in self.selection.items()}
        available=[v["mean_px"] for v in scores.values() if v["mean_px"] is not None]
        result=training.diagnostic_summary(self.baseline)
        result.update(coc_balanced_px=float(np.mean(available)) if len(available)==4 else None,coc_by_zone=scores,
                      component={k:stats(v) for k,v in self.values.items()},
                      saturation={k:dict(v,fraction=training.fraction(v["saturated"],v["n"]))
                                  for k,v in self.saturation.items()},
                      adjacent_switch=dict(self.switch,fraction=training.fraction(self.switch["changed"],
                                                                                self.switch["pairs"])))
        return result


@torch.inference_mode()
def diagnose(model, rows, selected):
    report=Diagnostics()
    for n,index in enumerate(selected):
        if n%5==0:
            training.gpu_gate(training.TRAIN_BUDGET[model.feature_mode])
        target=training.frame_batch(rows,[(index,0,0)],model.feature_mode,512)
        report.add(training.predict(model,target),target)
    result=report.summary()
    if result["coc_balanced_px"] is None or not math.isfinite(result["coc_balanced_px"]):
        raise ValueError("The four preregistered holdout zones must have finite CoC statistics")
    return result


def choose(curve):
    def rod(row):
        return row["holdout"]["errors"]["rod/decoded"]["p50"]
    eligible=[r for r in curve if rod(r)<=r["holdout"]["errors"]["rod/OVG"]["p50"]]
    if eligible:
        return min(eligible,key=lambda r:(r["holdout"]["coc_balanced_px"],r["step"])),True
    return min(curve,key=lambda r:(rod(r),r["step"])),False
