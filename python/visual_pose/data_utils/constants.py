import os
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).parent.parent.parent.parent
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