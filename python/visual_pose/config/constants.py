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
# Matterport's splits, one folder each: train is scenes 00000-00799, val 00800-00899.
# minival's 10 scenes are byte-identical copies of val's 00800-00809 (checked 2026-10-01).
HM3D_SPLITS = {
    "minival": REPO_DIR / "data" / "hm3d" / "hm3d-minival-habitat-v0.2",
    "val": REPO_DIR / "data" / "hm3d" / "hm3d-val-habitat-v0.2",
    "train": REPO_DIR / "data" / "hm3d" / "hm3d-train-habitat-v0.2",
}
# OUR roles, not Matterport's split names: role -> {split: scene numbers (the 5-digit
# prefix)}. The one thing to edit. Renderer, sample building, split_trajectories and
# the GUI all read it through hm3d_sequence.scene_dirs(role).
HM3D_SCENES = {
    "training": {"val": range(800, 900), "train": [*range(0, 100), *range(200, 331)]},
    "validation": {"train": range(100, 200)},
}
# Rendered per scene. 215 MB a trajectory on disk. Validation wants many scenes, not many
# paths, and every epoch runs all of it (a quarter of epoch time at 200 graphs).
HM3D_TRAJECTORIES_PER_SCENE = {"training": 10, "validation": 3}
# Every split some role uses; a scene name in several of them resolves to the first.
HM3D_ACTIVE = tuple(dict.fromkeys(split for role in HM3D_SCENES.values() for split in role))
HM3D_ROOTS = [HM3D_SPLITS[s] for s in HM3D_ACTIVE]


def _check_roles_disjoint() -> None:
    """A scene in both roles would put the same house in train and val."""
    for split in HM3D_ACTIVE:
        a, b = (set(HM3D_SCENES[r].get(split, ())) for r in ("training", "validation"))
        if a & b:
            raise ValueError(f"HM3D_SCENES: {split} scenes {sorted(a & b)[:5]}... in both roles")
_check_roles_disjoint()
