"""Run the unchanged author CLI with the RTX 5090 / xformers 0.0.32 compatibility fix."""
import os
from pathlib import Path
import runpy
import sys


def configure_attention():
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
        # The pickled ControlNet contains XFormersAttnProcessor instances.
        # xformers 0.0.32.post2 incorrectly selects its Hopper FA3 kernel here.
        # Keep the model/attention processors intact and use the remaining dispatch backends.
        from xformers.ops.fmha import _set_use_fa3
        _set_use_fa3(False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    repo = root.parent / "MagicBokeh"
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    configure_attention()
    # Preserve the author's relative paths; call from its repository directory.
    sys.path.insert(0, str(repo))
    runpy.run_path(str(repo / "inference.py"), run_name="__main__")
