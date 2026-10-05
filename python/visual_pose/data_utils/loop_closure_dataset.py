"""
Labelled HM3D keyframe pairs for training retrieval: true loop closures
against false ones, both from inside one trajectory (so every negative is the
same house -- the hard case).

Four cells per trajectory, up to PAIRS_PER_CELL each, so neither class can be
told apart by where it was drawn from (frontend candidates have high GeM
cosine; drawing positives only at random would teach "high cosine = false"):

                  frontend candidates     random eligible pairs
    true          hard positives          easy positives
    false         hard negatives          easy negatives

True is overlap > POSITIVE_OVERLAP in either direction, false < NEGATIVE_OVERLAP;
pairs between are dropped as ambiguous. Split by scene, never by trajectory.

Both caches are built ahead of training, labels on CPU then candidates on GPU:

    PYTHONPATH=. .venv/bin/python -m visual_pose.data_utils.loop_closure_dataset
"""
from dataclasses import replace
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from visual_pose.config import Config
from visual_pose.data_utils.loop_closures import CACHE_DIR, LoopClosures
from visual_pose.data_utils.sources import HM3DSource, SavedHM3DFrames, source

POSITIVE_OVERLAP = 0.3
NEGATIVE_OVERLAP = 0.05
PAIRS_PER_CELL = 64
VAL_EVERY = 5            # every 5th scene, in sorted order, is validation
CFG = Config(run_name="fine_tuning_wide_baseline")   # the frontend the hard cells come from


def trajectory_keys(split: str) -> list[str]:
    """hm3d: keys of every trajectory on disk in `split` ("train" or "val")."""
    return [k for k in HM3DSource.discover() if split_of(k) == split]


def split_of(key: str) -> str:
    """"train" or "val" for an hm3d key. Every descriptor and pooling run so far
    trained on "train" only, so Phase 3 must test on "val" scenes."""
    scenes = sorted({k.split("/")[0] for k in HM3DSource.discover()})
    return "val" if scenes.index(key.split("/")[0]) % VAL_EVERY == 0 else "train"


def loop_closures(key: str) -> LoopClosures:
    """Cached overlaps, keyed as in experiments/loop_candidate_stats.py, so either builds them for both."""
    return LoopClosures(source(key).frames(), replace(CFG, min_frame_gap=CFG.loop_min_gap),
                        name=f"{key}_kf1")


def _candidate_path(key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"loop_candidates_{safe}_{CFG.run_name}_{CFG.loop_selection}{CFG.loop_max_candidates}.npy"


def build_candidates(keys: list[str]) -> None:
    """(K, 2) frame pairs WorldModel.loop_candidates proposes, per trajectory. GPU."""
    from visual_pose.checkpoints import load_cnn_model
    from visual_pose.models.global_descriptor import gem_pooling
    from visual_pose.pose_graph.world_model import WorldModel
    model = load_cnn_model(CFG).eval()
    model.device = next(model.parameters()).device
    for n, key in enumerate(keys):
        path = _candidate_path(key)
        if path.exists():
            continue
        seq = source(key).frames()
        frames = list(range(len(seq)))       # HM3D frames are keyframes
        world = WorldModel(model, gem_pooling, seq, CFG, get_relative_pose=None)
        world.encode_keyframes(frames)
        np.save(path, np.array(world.loop_candidates(frames), dtype=np.int64).reshape(-1, 2))
        print(f"candidates {n + 1}/{len(keys)} {key}", flush=True)


def _build_labels(key: str) -> None:
    """Pool worker; returns nothing so the overlap matrix is not pickled back."""
    loop_closures(key)


def build_caches(keys: list[str]) -> None:
    # Processes for the labels: a Python loop of numpy geometry holds the GIL.
    # Two cores left free to keep the machine usable.
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        for n, _ in enumerate(pool.imap_unordered(_build_labels, keys)):
            print(f"labels {n + 1}/{len(keys)}", flush=True)
    build_candidates(keys)


class LoopClosureDataset(Dataset):
    """
    Yields rgb_i, rgb_j (3, H, W) float in [0, 1], label (1 true / 0 false),
    overlap (max of both directions), hard (drawn from frontend candidates),
    and traj, i, j: traj indexes self.overlaps, so cross-pair labels within a
    batch can be looked up for in-batch negatives.
    """

    def __init__(self, split: str = "train", pairs_per_cell: int = PAIRS_PER_CELL,
                 augment: bool = False, jitter: float = 0.2, seed: int = 0):
        self.keys = trajectory_keys(split)
        self.frames: list[SavedHM3DFrames] = []
        self.overlaps: list[np.ndarray] = []
        self.items: list[tuple[int, int, int, int, bool]] = []    # (traj, i, j, label, hard)
        for t, key in enumerate(self.keys):
            path = _candidate_path(key)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: build the caches first (see module docstring)")
            overlap = loop_closures(key).overlap
            overlap = np.maximum(overlap, overlap.T)
            self.frames.append(source(key).frames())
            self.overlaps.append(overlap)
            self.items += self._sample(t, overlap, np.load(path), np.random.default_rng(seed + t),
                                       pairs_per_cell)
        self.jitter = ColorJitter(brightness=jitter, contrast=jitter,
                                  saturation=jitter, hue=jitter / 8) if augment else None

    @staticmethod
    def _sample(t: int, overlap: np.ndarray, candidates: np.ndarray, rng: np.random.Generator,
                per_cell: int) -> list[tuple[int, int, int, int, bool]]:
        n = len(overlap)
        i, j = np.nonzero(np.triu(np.ones((n, n), dtype=bool), CFG.loop_min_gap))
        is_candidate = np.zeros((n, n), dtype=bool)
        is_candidate[candidates[:, 0], candidates[:, 1]] = True
        ov, hard = overlap[i, j], is_candidate[i, j]
        out = []
        for label, mask in ((1, ov > POSITIVE_OVERLAP), (0, ov < NEGATIVE_OVERLAP)):
            for h in (True, False):
                cell = np.nonzero(mask & (hard == h))[0]
                for k in rng.choice(cell, size=min(per_cell, len(cell)), replace=False):
                    out.append((t, int(i[k]), int(j[k]), label, h))
        return out

    def __len__(self) -> int:
        return len(self.items)

    def _image(self, t: int, i: int) -> Tensor:
        img = torch.from_numpy(self.frames[t].rgb(i)).permute(2, 0, 1).float() / 255.0
        return self.jitter(img) if self.jitter is not None else img   # separate draw per image

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        t, i, j, label, hard = self.items[index]
        return {
            "rgb_i": self._image(t, i),
            "rgb_j": self._image(t, j),
            "label": torch.tensor(label, dtype=torch.float32),
            "overlap": torch.tensor(self.overlaps[t][i, j], dtype=torch.float32),
            "hard": torch.tensor(hard),
            "traj": torch.tensor(t),
            "i": torch.tensor(i),
            "j": torch.tensor(j),
        }



class LoopClosureTrajectories(Dataset):
    """
    One item per trajectory, for InfoNCE with negatives mined from the model
    itself: the loss needs every frame of a house at once, both to find each
    anchor's hardest false partners and to tell which in-batch pairs are
    actually closures.

        rgb      (N, 3, H, W) uint8 -- float on the GPU, a 4x smaller transfer
        overlap  (N, N) float32, max of both directions; 0 under CFG.loop_min_gap
        frame    (N,) each row's keyframe index in the trajectory, for gaps
        traj     index into self.keys

    stride > 1 keeps every stride-th keyframe from a random offset: neighbours
    are 0.6 m / 13.5 deg apart and see nearly the same surfaces, so mining
    on half of them halves the CNN pass that dominates a step. Val uses 1,
    since the frontend retrieves over every keyframe.
    """

    def __init__(self, split: str = "train", keys: list[str] | None = None,
                 augment: bool = False, jitter: float = 0.2, stride: int = 1):
        self.keys = keys if keys is not None else trajectory_keys(split)
        self.stride = stride
        self.jitter = ColorJitter(brightness=jitter, contrast=jitter,
                                  saturation=jitter, hue=jitter / 8) if augment else None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        key = self.keys[index]
        frames = source(key).frames()
        offset = int(torch.randint(self.stride, ())) if self.stride > 1 else 0
        keep = np.arange(offset, len(frames), self.stride)
        rgb = torch.from_numpy(np.stack([frames.rgb(i) for i in keep])).permute(0, 3, 1, 2)
        if self.jitter is not None:          # one draw per frame, as the pair datasets do
            rgb = torch.stack([(self.jitter(x.float() / 255) * 255).round().to(torch.uint8) for x in rgb])
        overlap = loop_closures(key).overlap[np.ix_(keep, keep)]
        return {"rgb": rgb.contiguous(),
                "overlap": torch.from_numpy(np.maximum(overlap, overlap.T)),
                "frame": torch.from_numpy(keep),
                "traj": torch.tensor(index)}

if __name__ == "__main__":
    build_caches(trajectory_keys("train") + trajectory_keys("val"))
