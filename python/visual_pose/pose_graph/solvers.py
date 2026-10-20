import numpy as np
import torch
from torch import nn

from visual_pose import _geometry as cpp
from visual_pose.config.config import TransformerSolverConfig
from visual_pose.pose_graph.world_model import WorldModel


def gauss_newton_solver(
        world: WorldModel,
        iter_threshold: int,
        verbose: bool,
        huber: float,
        lm_lambda_init: float
) -> None:
    cpp.FactorGraph.gauss_newton(
        world.factor_graph,
        iter_threshold=iter_threshold,
        verbose=verbose,
        huber=huber,
        lm_lambda_init=lm_lambda_init
    )


def transformer_solver(world: WorldModel, *, model: nn.Module, cfg: TransformerSolverConfig,
                       descriptor_model=None) -> None:
    """
    The trained transformer as a graph solver, poses written back into world.factor_graph. The
    unsolved graph goes through training's own path -- sample_arrays -> (descriptors_at) ->
    model_item -> collate -> refine -- so the model sees what it was trained on: same features,
    points drawn with the same rng, the first points_per_keyframe kept as at validation.
    descriptor_model: the SemanticDescriptor the samples' descriptors came from, when cfg.use_descriptor.
    Needs the RAW information (world.optimize(..., calibrate=False)) and the unsolved graph
    (vertex 0 = identity).
    """
    # here, not at the top: these pull in the training stack, which the GN-only path never needs
    from visual_pose.config.constants import DEVICE
    from visual_pose.data_utils.loaders import collate_pose_graph_ds
    from visual_pose.data_utils.pose_graph_dataset import model_item
    from visual_pose.data_utils.pose_graph_samples import descriptors_at, sample_arrays
    from visual_pose.evaluation.solver_comparison import refine
    vertex_frames = np.array(sorted(world.vertex_index, key=world.vertex_index.get))
    data = sample_arrays(world.factor_graph, np.array(world.edge_is_closure), vertex_frames, world.sequence,
                         cfg.num_point_samples)
    keep = slice(0, cfg.points_per_keyframe)
    descriptors = None
    if cfg.use_descriptor:
        if descriptor_model is None:
            raise ValueError("transformer_solver: cfg.use_descriptor needs descriptor_model (SemanticDescriptor)")
        uvs = torch.from_numpy(data["uvs"][:, keep]).int().to(DEVICE)
        # float16 as the descriptor pass stores them, so the model sees the same rounding as in training
        descriptors = descriptors_at(world.sequence, vertex_frames, descriptor_model, uvs)[0].half()
    # closures as at validation: none for a model trained without them (closure_dropout 1.0)
    item = model_item(data, keep, descriptors, closure_dropout=1.0 if cfg.closure_dropout == 1.0 else 0.0)
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in collate_pose_graph_ds([item]).items()}
    T, _ = refine(model, batch, data["poses"], cfg)
    world.factor_graph.set_poses(list(T))
