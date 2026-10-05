"""
The attention-pooling aggregator's pieces for training.fit. The CNN is frozen
and bound in by partial (see train.train_atn_pooling), so `model` is only the
aggregator. A batch is one LoopClosureTrajectories item, on DEVICE:

    rgb      (1, N, 3, H, W) uint8, the trajectory's keyframes (every stride-th)
    overlap  (1, N, N) max of both directions; 0 under loop_min_gap
    frame    (1, N) each row's keyframe number

cnn(images) returns (descriptors, backbone); the aggregator pools the backbone,
(B, 256, 60, 80) -- what GeM pools, and the stronger input for retrieval (00803:
AUC 0.839 and top-500 28.2% against 0.793 and 22.6% pooling the descriptor head).
"""
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE
from visual_pose.data_utils.loaders import on_device
from visual_pose.models.global_descriptor import gem_pooling
from visual_pose.pose_graph.world_model import WorldModel

CHUNK = 32            # frames per CNN / aggregator pass when scoring a whole trajectory


def backbone_maps(cnn: nn.Module, rgb: Tensor) -> Tensor:
    """(N, 3, H, W) uint8 -> (N, 256, H', W') half: 2.4 MB a frame, ~150 MB a stride-4 step."""
    with torch.no_grad():
        rgb = rgb.to(next(cnn.parameters()).device)
        return torch.cat([cnn(rgb[s:s + CHUNK].float().div_(255))[1].half()
                          for s in range(0, len(rgb), CHUNK)])


def embed(model: nn.Module, maps: Tensor) -> Tensor:
    """(B, C, H', W') backbone maps -> (B, D) unit fingerprints; the model tokenises."""
    maps = maps.to(next(model.parameters()).device)
    return F.normalize(model(maps.float()), dim=-1)


def fingerprints(model: nn.Module, maps: Tensor) -> Tensor:
    with torch.no_grad():
        return torch.cat([embed(model, maps[s:s + CHUNK]) for s in range(0, len(maps), CHUNK)])


def aggregator_step(model: nn.Module, batch: dict[str, Tensor], cnn: nn.Module, cfg: Config) -> Tensor:
    """
    InfoNCE over cfg.aggregator_anchors anchors of one trajectory. Row k scores anchor k
    against every column: each anchor's positive (one random true partner), and
    each anchor's cfg.aggregator_hard_negatives highest-scoring false partners under the model
    as it is now -- its own mistakes, re-mined every step. Mining from GeM's
    candidates once taught the model to fix GeM's errors, and its own top 500
    came out no purer (51.0% vs GeM's 51.5%).

        L_k = -log[ exp(s_kk / t) / sum_c exp(s_kc / t) ],  s = cosine

    A column enters row k's denominator only if it is a false partner of
    anchor k: another anchor's positive can be a true closure of this one, and
    a frame under loop_min_gap is neither (retrieval never proposes it).
    """
    # the CNN pass is ~75% of a step and its maps never change (frozen), so
    # repeated steps on one batch (fit steps_per_batch) reuse them
    if "maps" not in batch:
        batch["maps"] = backbone_maps(cnn, batch["rgb"][0])
    maps = batch["maps"]
    overlap = batch["overlap"][0]
    frame = batch["frame"][0]                      # keyframe numbers: gaps stay in keyframes under a stride
    eligible = (frame[None] - frame[:, None]).abs() >= cfg.loop_min_gap
    true = eligible & (overlap > cfg.loop_true_overlap)
    false = eligible & (overlap <= cfg.loop_true_overlap)

    candidates = true.any(dim=1).nonzero().squeeze(1)
    if len(candidates) == 0:                       # a trajectory with no revisit: nothing to learn
        return sum(p.sum() for p in model.parameters()) * 0.0
    anchors = candidates[torch.randperm(len(candidates), device=overlap.device)[:cfg.aggregator_anchors]]
    positives = torch.multinomial(true[anchors].float(), 1).squeeze(1)
    current = fingerprints(model, maps)                                           # (N, D), no grad
    scores = current[anchors] @ current.T                                         # (K, N)
    hard = scores.masked_fill(~false[anchors], -torch.inf).topk(cfg.aggregator_hard_negatives, dim=1).indices

    columns = torch.cat([positives, hard.flatten()])
    frames = torch.unique(torch.cat([anchors, columns]))      # sorted, so searchsorted indexes it
    e = embed(model, maps[frames])                            # the only pass with gradients
    e_anchor = e[torch.searchsorted(frames, anchors)]
    e_column = e[torch.searchsorted(frames, columns)]

    k = len(anchors)
    keep = false[anchors][:, columns]
    keep[torch.arange(k), torch.arange(k)] = True             # each row's own positive
    logits = (e_anchor @ e_column.T / cfg.aggregator_temperature).masked_fill(~keep, -torch.inf)
    return F.cross_entropy(logits, torch.arange(k, device=logits.device))


class SigmoidScaled(nn.Module):
    """
    A pooling model plus the sigmoid loss's two learned scalars, so the
    optimiser and the checkpoint carry them; forward is the pooling model's.
    load_learned_pooling strips the wrapper.
    """

    def __init__(self, pool: nn.Module, scale: float = 10.0, bias: float = 11.5):
        super().__init__()
        self.pool = pool
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        # t c - b near logit(0.12) at the untrained cosines (~0.95) and base rate (~12% true)
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


def aggregator_sigmoid_step(model: SigmoidScaled, batch: dict[str, Tensor], cnn: nn.Module, cfg: Config) -> Tensor:
    """
    Pairwise sigmoid loss (SigLIP) over EVERY eligible pair of the trajectory,
    one learned scale t and threshold b shared by all of them:

        L = mean over pairs  -log sigma( y (t c - b) ),   y = +1 true, -1 false

    One threshold for all pairs is what top-500 selection over the whole
    trajectory needs. InfoNCE ranks each anchor's row on its own and is blind
    to a shift of a whole row, so a frame similar to everything can fill the
    global list while every row looks right.
    """
    if "maps" not in batch:
        batch["maps"] = backbone_maps(cnn, batch["rgb"][0])
    overlap, frame = batch["overlap"][0], batch["frame"][0]
    eligible = torch.triu((frame[None] - frame[:, None]).abs() >= cfg.loop_min_gap, diagonal=1)
    i, j = eligible.nonzero(as_tuple=True)
    if len(i) == 0:                                # too short for any pair loop_min_gap apart
        return sum(p.sum() for p in model.parameters()) * 0.0
    e = embed(model, batch["maps"])                # every frame, with gradients
    y = torch.where(overlap[i, j] > cfg.loop_true_overlap, 1.0, -1.0)
    logits = model.log_scale.exp() * (e[i] * e[j]).sum(-1) - model.bias
    return F.softplus(-y * logits).mean()


def selection_precision(model: nn.Module, loader: DataLoader, cnn: nn.Module, cfg: Config) -> dict:
    """
    (fingerprint, mode) -> precision of the frontend's own candidate selection
    (span or topk, loop_max_candidates, pairs at least loop_min_gap apart),
    pooled over the loader's trajectories, for the learned fingerprint and GeM.
    """
    totals = {(name, mode): [0, 0] for name in ("learned", "GeM") for mode in ("span", "topk")}
    for batch in on_device(loader, DEVICE):
        rgb, true = batch["rgb"][0], (batch["overlap"][0] > cfg.loop_true_overlap).cpu().numpy()
        learned, gem = [], []
        with torch.no_grad():
            for s in range(0, len(rgb), CHUNK):
                _, backbone = cnn(rgb[s:s + CHUNK].float().div_(255))
                learned.append(embed(model, backbone).cpu())
                gem.append(gem_pooling(backbone).cpu())
        n = len(rgb)
        for name, v in (("learned", torch.cat(learned)), ("GeM", torch.cat(gem))):
            similarity = (v @ v.T).cpu().numpy()
            # each as the frontend runs it: GeM at loop_retrieval_similarity
            # (admits ~every pair; 00800 gives identical picks at 0.94 and -1),
            # the learned one at -1, since its cosines sit far below 0.94
            threshold = cfg.loop_retrieval_similarity if name == "GeM" else -1.0
            pairs = [(i, j) for i in range(n) for j in range(i + cfg.loop_min_gap, n)
                     if similarity[i, j] >= threshold]
            for mode in ("span", "topk"):
                chosen = WorldModel._select(SimpleNamespace(cfg=replace(cfg, loop_selection=mode)),
                                            pairs, similarity, cfg.loop_max_candidates)
                totals[name, mode][0] += int(sum(true[a, b] for a, b in chosen))
                totals[name, mode][1] += len(chosen)
    return {key: hit / max(n_chosen, 1) for key, (hit, n_chosen) in totals.items()}


def aggregator_evaluate(model: nn.Module, val_loader: DataLoader, cnn: nn.Module, cfg: Config,
                        train_probe: DataLoader | None = None) -> float:
    """
    selection_precision on the val trajectories: learned span is the checkpoint
    metric, GeM printed beside it as the real bar. With train_probe, the same on
    a fixed handful of TRAINING trajectories: train far above val means the
    model memorises houses; both flat means it is not learning at all.
    """
    for name, loader in (("val", val_loader), ("train", train_probe)):
        if loader is None:
            continue
        p = selection_precision(model, loader, cnn, cfg)
        print(f"  {name:5s} top-{cfg.loop_max_candidates} precision  " + "  ".join(
            f"{fp} {mode} {v:.1%}" for (fp, mode), v in p.items()))
        if name == "val":
            value = p["learned", "span"]
    return value
