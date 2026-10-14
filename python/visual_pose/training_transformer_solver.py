import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from visual_pose.config import TransformerSolverConfig
from visual_pose.data_utils.loaders import on_device
from visual_pose.geometry.graph_math import relative, predicted_poses, update_graph_with_tangent_delta, \
    update_ground_truth_residual
from visual_pose.training import EMPTY_CACHE_EVERY, empty_cache
from roma import rotvec_to_rotmat


# batch keys passed to TransformerSolver.forward, same names. Targets (gt_poses,
# delta_target) must never be in here.
MODEL_INPUTS = [
    "poses",
    "points",
    "closures",
    "closures_ij",
    "odometry",
    "odometry_ij",
    "descriptors",
    "vertex_mask",
    "closure_mask",
    "odom_mask",
]

def huber(error: Tensor, threshold: float) -> Tensor:
    """Elementwise Huber of a non-negative distance: 0.5 e^2 below threshold, linear above (pull capped at threshold)."""
    return torch.where(error.abs() < threshold, 0.5 * error ** 2, threshold * (error.abs() - 0.5 * threshold))


def masked_mean(err: Tensor, padding: Tensor) -> Tensor:
    """Mean of err (B, N) over real entries; padding (B, N) bool, True = padding. 0 when nothing is real."""
    real = (~padding).float()                                    # (B, N)
    return (err * real).sum() / real.sum().clamp(min=1)         # clamp: a batch can have no closures at all


def pose_distance(pred_R: Tensor, pred_t: Tensor, target_R: Tensor, target_t: Tensor,
                  cfg: TransformerSolverConfig) -> Tensor:
    """
    Per item, chordal rotation error + weighted Huber translation error.
    R (..., 3, 3), t (..., 3) -> (...). ||R - R*||_F^2 = 4 (1 - cos theta): no wrap at 180 deg, at most 8.
    """
    rot_err = torch.linalg.matrix_norm(pred_R - target_R) ** 2      # (...)
    trans_err = torch.linalg.norm(pred_t - target_t, dim=-1)         # (...) metres
    return rot_err + cfg.loss_trans_weight * huber(trans_err, cfg.loss_huber_threhold)


def vertex_loss(predicted_delta: Tensor, batch: dict[str, Tensor], cfg: TransformerSolverConfig) -> Tensor:
    """
    Each vertex's predicted correction against the true one, T_in^-1 T_gt (delta_target_mat
    (B, V, 4, 4)). Absolute: mostly rewards removing global drift, not shape.
    """
    target = batch["delta_target_mat"]                               # (B, V, 4, 4)
    err = pose_distance(rotvec_to_rotmat(predicted_delta[..., 3:]), predicted_delta[..., :3],
                        target[..., :3, :3], target[..., :3, 3], cfg)  # (B, V)
    return masked_mean(err, batch["vertex_mask"])


def relative_loss(predicted_delta: Tensor, batch: dict[str, Tensor], cfg: TransformerSolverConfig) -> Tensor:
    """
    Predicted against true relative pose on every odometry and closure edge, each a masked
    mean, summed. Gauge-free: a global correction cannot lower it, only getting the shape right can.
    """
    pred_R, pred_t = predicted_poses(predicted_delta, batch["poses"])        # (B, V, 3, 3), (B, V, 3)
    gt_R, gt_t = batch["gt_poses"][..., :3, :3], batch["gt_poses"][..., :3, 3]
    loss = predicted_delta.new_zeros(())
    for ij_key, mask_key, weight in (("odometry_ij", "odom_mask", cfg.loss_relative_odom_weight), ("closures_ij", "closure_mask", cfg.loss_relative_closure_weight)):
        ij = batch[ij_key]                                           # (B, E, 2)
        err = pose_distance(*relative(pred_R, pred_t, ij), *relative(gt_R, gt_t, ij), cfg)  # (B, E)
        loss += weight * masked_mean(err, batch[mask_key])
    return loss


def weighted_l2_loss(predicted_delta: Tensor, batch: dict[str, Tensor], cfg: TransformerSolverConfig) -> Tensor:
    """L2 on tangent-space delta vs delta_target (B, V, 6). Superseded: Log's cut at 180 deg, unbounded."""
    err = predicted_delta - batch['delta_target']                    # (B, V, 6)
    err[..., :3] *= cfg.loss_trans_weight
    return masked_mean((err ** 2).sum(dim=-1), batch['vertex_mask'])


def pose_loss(predicted_delta: Tensor, batch: dict[str, Tensor], cfg: TransformerSolverConfig) -> Tensor:
    """cfg.loss_fn: "l2" | "chordial" (vertex only) | "chordial_relative" (vertex + loss_relative_weight * relative)."""
    if cfg.loss_fn == "l2":
        return weighted_l2_loss(predicted_delta, batch, cfg)
    if cfg.loss_fn == "chordial":
        return vertex_loss(predicted_delta, batch, cfg)
    if cfg.loss_fn == "chordial_relative":
        return (vertex_loss(predicted_delta, batch, cfg)
                + relative_loss(predicted_delta, batch, cfg))
    raise ValueError(f"Unknown loss function: {cfg.loss_fn}")


def trans_step(model: nn.Module, batch: dict[str, Tensor], cfg: TransformerSolverConfig,
               random_depth: bool = True) -> Tensor:
    """
    fit's step(model, batch), cfg bound by partial in train.py: masks come in the batch from
    collate_pose_graph_ds. k refinement passes without gradients, then one trained pass whose
    loss is returned: memory is one pass whatever k is, since no_grad keeps no activations.
    random_depth: k drawn from 0..refinement_iterations-1 each step, so the trained pass sees
    raw odometry (k = 0) as well as already-refined poses -- every input it meets at inference.
    Without it, it only ever trains on k = refinement_iterations-1 and can drift away from raw
    inputs. Evaluation passes False: always the full refinement, as at inference.
    """
    if cfg.refinement_backprop and random_depth:
        # Training alternative: every pass in the loss, gradients through all of them (unrolled,
        # like RAFT). Memory ~ refinement_iterations passes; evaluation still takes the path below.
        loss = 0.0
        for _ in range(cfg.refinement_iterations):
            pred = model(**{k: batch[k] for k in MODEL_INPUTS})   # (B, V, 6)
            loss = loss + pose_loss(pred, batch, cfg)
            batch['poses'] = update_graph_with_tangent_delta(batch['poses'], pred)
            batch['delta_target_mat'], batch['delta_target'] = update_ground_truth_residual(batch['poses'], batch['gt_poses'])
        return loss / cfg.refinement_iterations
    k = (int(torch.randint(cfg.refinement_iterations, ())) if random_depth
         else cfg.refinement_iterations - 1)
    train = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(k):
            pred = model(**{k: batch[k] for k in MODEL_INPUTS})   # (B, V, 6)
            batch['poses'] = update_graph_with_tangent_delta(batch['poses'], pred)
            batch['delta_target_mat'], batch['delta_target'] = update_ground_truth_residual(batch['poses'], batch['gt_poses'])
    if train:
        model.train()
    pred = model(**{k: batch[k] for k in MODEL_INPUTS})  # (B, V, 6)
    return pose_loss(pred, batch, cfg)


def trans_evaluate(model: nn.Module, val_loader: DataLoader, cfg: TransformerSolverConfig) -> float:
    """fit's evaluate(model, val_loader), cfg bound by partial: -(mean val loss over batches), since fit keeps the highest."""
    device = next(model.parameters()).device
    losses = []
    for i, batch in enumerate(on_device(val_loader, device)):
        losses.append(trans_step(model, batch, cfg, random_depth=False).item())
        # the val pass alone grew the MPS cache to 29 GB (variable batch shapes)
        if (i + 1) % EMPTY_CACHE_EVERY == 0:
            empty_cache()
    return -sum(losses) / len(losses)


