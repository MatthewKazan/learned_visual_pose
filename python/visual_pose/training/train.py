"""
Training entrypoint. Every tunable lives in visual_pose.config.config.Config.

    python -m visual_pose.training.train
    python -m visual_pose.training.train --config-json overrides.json
"""
import argparse
import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import torch

from visual_pose.training.checkpoints import load_cnn_model
from visual_pose.config.config import Config, TransformerSolverConfig
from visual_pose.data_utils.loaders import (build_loaders, build_loop_closure_probe_loader,
                                            build_loop_closure_trajectory_loaders, build_transformer_loaders)
from visual_pose.models.descriptor_cnn import DescriptorCNN
from visual_pose.models.global_descriptor import AttentionPooling2d
from visual_pose.models.transformer_solver.transformer import TransformerSolver
from visual_pose.training.fit import fit
from visual_pose.training.cnn import cnn_evaluate, cnn_step
from visual_pose.training.aggregator import (SigmoidScaled, aggregator_evaluate, aggregator_sigmoid_step,
                                             aggregator_step)
from visual_pose.training.transformer_solver import trans_evaluate, trans_step


def train_cnn(cfg: Config) -> None:
    # seeds torch's RNG before weight init and before the DataLoader derives
    # per-worker seeds from it, so grid offsets and subsampling replay too
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = build_loaders(cfg)
    fit(DescriptorCNN.from_config(cfg), train_loader, val_loader,
        step=partial(cnn_step, temperature=cfg.temperature),
        evaluate=partial(cnn_evaluate, tau=cfg.checkpoint_tau),
        settings=cfg.train_settings(), metric_name=f"val MMA@{cfg.checkpoint_tau}",
        config=asdict(cfg))


def train_atn_pooling(cfg: Config) -> None:
    """AttentionPooling over the frozen CNN's descriptor map; step and evaluate are in training/aggregator.py."""
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = build_loop_closure_trajectory_loaders(cfg)
    cnn = load_cnn_model(replace(cfg, run_name=cfg.aggregator_frozen_cnn)).eval()
    cnn.requires_grad_(False)
    pooling = AttentionPooling2d(cfg.body_channels[-1])                    # pools the backbone
    if cfg.aggregator_loss == "sigmoid":
        model, step = SigmoidScaled(pooling), aggregator_sigmoid_step
    elif cfg.aggregator_loss == "infonce":
        model, step = pooling, aggregator_step
    else:
        raise ValueError(f"unknown aggregator_loss {cfg.aggregator_loss!r}, expected sigmoid|infonce")
    fit(model, train_loader, val_loader,
        step=partial(step, cnn=cnn, cfg=cfg),
        evaluate=partial(aggregator_evaluate, cnn=cnn, cfg=cfg,
                         train_probe=build_loop_closure_probe_loader(cfg)),
        settings=cfg.aggregator_train_settings(),
        metric_name=f"val span precision@{cfg.loop_max_candidates}",
        config=asdict(cfg),
        steps_per_batch=cfg.aggregator_draws_per_trajectory, evaluate_before=True)

def train_transformer_solver(cfg: TransformerSolverConfig | None = None) -> None:
    cfg = cfg or TransformerSolverConfig()
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = build_transformer_loaders(cfg)
    model = TransformerSolver.from_config(cfg)
    from visual_pose.evaluation.solver_comparison import ate_epoch_hook
    fit(model, train_loader, val_loader,
        step=partial(trans_step, cfg=cfg),          # fit passes the model and batch
        evaluate=partial(trans_evaluate, cfg=cfg),
        settings=cfg.train_settings(), metric_name="-val loss",
        config=asdict(cfg), evaluate_before=True,
        on_epoch_end=ate_epoch_hook(model, cfg))

def main(cfg: Config | None = None) -> None:
    # --model picks what to train; --config-json lets an orchestrator hand over a
    # config (e.g. a sweep winner) without a flag per field. Anything absent from
    # the file keeps its default.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("transformer", "aggregator"), default="transformer")
    ap.add_argument("--config-json", default=None,
                    help="JSON dict of config overrides")
    args = ap.parse_args()
    overrides = json.loads(Path(args.config_json).read_text()) if args.config_json else {}
    if args.model == "transformer":
        train_transformer_solver(TransformerSolverConfig(**overrides))
        return
    if cfg is None:
        # JSON has no tuples, so tuple-typed fields come back as lists
        overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        cfg = Config(**overrides)
    print(f"run: {cfg.run_name}\n{cfg.summary()}\n")
    train_atn_pooling(cfg)


if __name__ == "__main__":
    main()
