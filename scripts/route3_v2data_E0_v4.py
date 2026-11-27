"""Train E0 v4 with the v3 recipe and without single-layer rod supervision."""
import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
from zoneinfo import ZoneInfo

import torch

import route3_v2data_E0_train as training
import route3_v2data_E0_v3 as v3
from route3_v2data_E0_tears import train as train_tear
from route3_v2data_E_labels import load, save_json
from route3_v2data_E0_v3_metrics import choose, diagnose

RUN = v3.ROOT / "outputs/route3-v2data-E0-v4-2026-11-27-01"
RECORD = v3.CONFIG.with_name("route3-v2data-E0-v4-2026-11-27.json")
CURVE_STEPS = {"v2": (2000, 4000, 6000, 8000), "v4": (2000, 4000, 6000, 8000)}


@torch.inference_mode()
def qualify():
    if RECORD.is_file():
        return load(RECORD)["step_zero"]
    choice, receipt, order, _, _ = v3.input_rows()
    if not choice["sha256"].startswith("dfb1d7e5"):
        raise ValueError("E0 v4 uses the registered frozen geometry")
    RUN.mkdir(parents=True, exist_ok=True)
    if not (RUN / "data.json").is_file():
        (RUN / "data.json").symlink_to((v3.RUN / "data.json").resolve())
    v3.setup()
    batch = training.frame_batch(receipt["rows"], order[0], v3.MODE)
    network = training.module(v3.MODE, scheme_a=True).eval()
    pred = training.predict(network, batch)
    eligible = batch["visible"] & ~batch["valid_dual"]
    by_zone = {}
    for index, name in enumerate(training.GROUPS):
        visible = eligible & (batch["loss_zone"] == index)
        scoped = dict(batch, visible=visible)
        values = [float(training.single_back_loss(pred, scoped, exclude_rod=exclude))
                  for exclude in (False, True)]
        if not torch.isfinite(torch.as_tensor(values)).all() or values[1] != (0. if index == 0 else values[0]):
            raise ValueError(f"Step-zero mask comparison failed for {name}: {values}")
        by_zone[name] = dict(pixels=int(visible.sum()), v3=values[0], v4=values[1])
    if not by_zone["rod"]["pixels"] or by_zone["rod"]["v3"] <= 0:
        raise ValueError("First training batch does not exercise the removed rod loss")
    full = {v: float(training.single_back_loss(pred, batch, exclude_rod=exclude))
            for v, exclude in (("v3", False), ("v4", True))}
    zero = dict(initial_batch="registered crop-order batch 0", by_zone=by_zone, **full)
    cfg = deepcopy(load(v3.CONFIG))
    cfg.pop("reachability", None)
    cfg.pop("old_selected_steps", None)
    cfg["step_zero"] = zero
    cfg["scheme_A"].update(
        support="visible & ~valid_dual & (loss_zone != 0)",
        objective="Sum of the gap, normal and boundary masked pixel means; rod contributes zero.",
        semantics="Single-surface rear targets use the v3 order constraint in gap, normal and boundary. "
                  "The single-rear term omits rod pixels. The dual-surface term retains the v3 support.",
    )
    cfg.update(status="registered",
               registered_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
               previous="route3-v2data-E0-v3-config-2026-11-26.json",
               geometry=choice, optical_curve=dict(steps=CURVE_STEPS, tear=False, selection=False),
               evaluation=dict(stage="E0 v4", development_cases=87, legacy_cases=18, test_cases=0,
                               focus=["P","G"], baseline="B (v1@2k + tear) from 3e135e5",
                               success="G F2 and S2 each have paired median >= 0 dB and pooled PSNR >= B."))
    save_json(RECORD, cfg)
    print("E0_V4_STEP_ZERO", zero, flush=True)
    return zero


def fit():
    return training.train(v3.MODE, run=RUN, config_path=RECORD, normal_front_weight=.1,
                          scheme_a=True, exclude_rod=True, diagnose_fn=diagnose, choose_fn=choose)


def tear():
    selected = load(RUN / v3.MODE / "training.json")["selected"]["step"]
    return train_tear(v3.MODE, selected, run=RUN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qualify", "train", "tear"))
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / f"{args.stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"qualify": qualify, "train": fit, "tear": tear}[args.stage]()
