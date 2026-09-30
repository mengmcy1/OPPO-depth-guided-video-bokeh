"""Run the unchanged author CLI with the RTX 5090 / xformers 0.0.32 compatibility fix."""
import os
from pathlib import Path
import runpy
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines.magicbokeh import configure_attention


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    repo = root.parent / "MagicBokeh"
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    configure_attention()
    # Preserve the author's relative paths; call from its repository directory.
    sys.path.insert(0, str(repo))
    runpy.run_path(str(repo / "inference.py"), run_name="__main__")
