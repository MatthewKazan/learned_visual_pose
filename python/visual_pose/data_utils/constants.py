import os
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).parent.parent.parent.parent


def code_version() -> str:
    """git describe --always --dirty, for provenance in generated files; never raises."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_DIR), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:   # noqa: BLE001 -- provenance must never abort generation
        return "unknown"


INTRINSICS_TARTAN_AIR = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def _pick_device() -> torch.device:
    # TORCH_DEVICE=cpu is the escape hatch: two processes holding Metal contexts
    # at once has already aborted a training run mid-epoch, so anything run
    # alongside a live training job should force CPU.
    if "TORCH_DEVICE" in os.environ:
        return torch.device(os.environ["TORCH_DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
DEVICE = _pick_device()

TARTAN_AIR_PATH = REPO_DIR / "data" / "tartanair"
HM3D_SPLITS = {
    "minival": REPO_DIR / "data" / "hm3d" / "hm3d-minival-habitat-v0.2",
    "val": REPO_DIR / "data" / "hm3d" / "hm3d-val-habitat-v0.2",
}
# The one line to edit. minival's 10 scenes are byte-identical copies of val's
# 00800-00809 (checked 2026-10-01), so a scene name can be in several splits:
# it resolves to the first split listed here, and the others' copy is unseen.
HM3D_ACTIVE = ("val",)
HM3D_ROOTS = [HM3D_SPLITS[s] for s in HM3D_ACTIVE]
