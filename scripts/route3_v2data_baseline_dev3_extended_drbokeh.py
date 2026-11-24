"""B2 Oracle-input Dr.Bokeh worker, using the pinned MagicBokeh environment."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
import torch

from route3_v2data_baseline_dev3 import ROOT, read
from src.baselines.drbokeh import OfficialDrBokeh, map_geometry
from src.baselines.drbokeh_cuda import load_grid2d


def run_request(adapter, request):
    with np.load(request["input"]) as data:
        rgb, depth, alpha = (data[k] for k in ("rgb","depth","alpha"))
    if not (np.isfinite(depth).all() and (depth > 0).all()):
        raise ValueError("The B2 Oracle requires positive known depths and 10000 m at no-hit pixels")
    inverse = 1/depth.astype(np.float64)
    rendered = {}
    audit = None
    for key,radius in (("rgb",request["radius"]),("zero",0.)):
        d, f, strength, mapping = map_geometry(
            inverse,1/request["focus"],request["fx"]*radius)
        value, original, _ = adapter.predict(rgb,d,f,strength,alpha_override=alpha[...,None])
        if original["calls"]["LDF"] or original["calls"]["MiDaS"]:
            raise ValueError("B2 Oracle accidentally invoked depth or segmentation")
        rendered[key] = np.clip(value,0,1).astype(np.float32)
        if audit is None:
            audit = dict(mapping=mapping,lens=adapter.protocol["lens"],calls=original["calls"],
                         max_lens_omitted_mass=max(v["radial_tail_mass_at_max"]
                                                   for v in original["actual_layers"].values()))
    np.savez(request["output"],**rendered)
    return audit


def main():
    output = sys.stdout

    def reply(value):
        print(json.dumps(value,allow_nan=False),file=output,flush=True)

    try:
        with redirect_stdout(sys.stderr):
            torch.set_num_threads(2)
            torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
            protocol = read(ROOT/"configs/drbokeh-track-a-frozen.json")
            source = ROOT/"outputs/route3-b2only-2026-11-10-01/manifests/oracle.json"
            protocol.update(lens=read(source)["drbokeh_lens"],oracle_depth=True,oracle_alpha=True)
            sys.modules["scatter_cuda"] = load_grid2d()
            adapter = OfficialDrBokeh(ROOT.parent/"DrBokeh",protocol,instrument=True)
    except Exception as error:
        reply(dict(error=f"{type(error).__name__}: {error}"))
        raise
    reply(dict(ready=True))
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):
                audit = run_request(adapter,json.loads(line))
            reply(dict(audit=audit))
        except Exception as error:
            if isinstance(error,torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            reply(dict(error=f"{type(error).__name__}: {error}"))


if __name__ == "__main__":
    main()
