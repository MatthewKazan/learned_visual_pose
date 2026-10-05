"""
The epoch loop, shared by every model trained here. What differs per model is
two functions, passed in:

    step(model, batch) -> loss             one batch's scalar loss
    evaluate(model, val_loader) -> metric  the checkpoint metric, higher is better

Anything frozen (the pretrained CNN under an aggregator) stays out of `model`
and is bound into step/evaluate with partial: then model.train() cannot flip
its BatchNorm, the optimiser never sees it, and best.pt holds only what was
trained.
"""
import json
from collections.abc import Callable

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from visual_pose.checkpoints import save_checkpoint
from visual_pose.config import TrainSettings
from visual_pose.data_utils.constants import DEVICE
from visual_pose.data_utils.loaders import on_device

MAX_SKIPPED_STEPS = 5   # consecutive non-finite gradients before fit gives up

StepFn = Callable[[nn.Module, dict[str, Tensor]], Tensor]
EvalFn = Callable[[nn.Module, DataLoader], float]


def make_optimizer(model: nn.Module, s: TrainSettings) -> tuple[Optimizer, LRScheduler]:
    """
    Optimizer over model's trainable parameters, cosine-annealed.

    Cosine rather than StepLR: StepLR's total decay is step_size times gamma,
    which desyncs from num_epochs and froze the last third of a run at lr/1000.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if s.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=s.learning_rate, momentum=s.momentum,
                              weight_decay=s.weight_decay)
    elif s.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=s.learning_rate, weight_decay=s.weight_decay)
    else:
        raise ValueError(f"unknown optimizer {s.optimizer!r}, expected sgd|adamw")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=s.num_epochs, eta_min=s.learning_rate * s.min_lr_factor)
    return opt, scheduler


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        step: StepFn, evaluate: EvalFn, settings: TrainSettings, *,
        metric_name: str = "val metric", config: dict | None = None,
        on_epoch_end: Callable[[int, float], None] | None = None,
        device: torch.device = DEVICE, steps_per_batch: int = 1,
        evaluate_before: bool = False) -> nn.Module:
    """
    Train `model`, evaluating every settings.eval_every epochs. Returns the model.

    Seed before building the model and loaders, not here: weight init has
    already happened by the time this runs. `config` is written beside the
    checkpoints as config.json (settings if None), so a result is always
    traceable. on_epoch_end lets a sweep report progress or abandon a trial by
    raising. `device` is where model and batches live; a step can still move
    work elsewhere (the aggregator runs its frozen CNN on the GPU).

    steps_per_batch > 1 takes that many optimizer steps on each batch; the step
    may cache its expensive inputs on the batch dict to reuse them.
    evaluate_before scores the untrained model first and saves it as best.pt,
    so the log starts with the baseline and best.pt can never be worse than it.
    """
    s = settings
    print(f"train {len(train_loader.dataset)} items / {len(train_loader)} batches, "
          f"val {len(val_loader.dataset)} items\n")
    model.to(device)
    optimizer, scheduler = make_optimizer(model, s)

    s.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (s.checkpoint_dir / "config.json").write_text(
        json.dumps(config if config is not None else s.__dict__, indent=2, default=list))

    best = -float("inf")
    skipped = 0
    if evaluate_before:
        model.eval()
        with torch.no_grad():
            best = evaluate(model, val_loader)
        save_checkpoint(s.checkpoint_dir / "best.pt", model, optimizer, scheduler, -1, best, metric_name)
        print(f"before training {metric_name}: {best:.4f}")
    for epoch in range(s.num_epochs):
        model.train()
        running = 0.0
        for batch_i, batch in enumerate(on_device(train_loader, device)):
            for _ in range(steps_per_batch):
                loss = step(model, batch)
                # one non-finite step poisons every weight through the optimizer, and
                # the run then trains on NaN for the rest of the epoch unnoticed
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, batch {batch_i + 1}")
                optimizer.zero_grad()
                loss.backward()
                # MPS has returned NaN gradients from finite inputs (training_aggregator);
                # skipping that step keeps the weights clean, a run of them means a real bug
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    skipped += 1
                    print(f"  non-finite gradient at epoch {epoch + 1}, batch {batch_i + 1}: step skipped ({skipped} in a row)")
                    if skipped >= MAX_SKIPPED_STEPS:
                        raise FloatingPointError(f"{skipped} non-finite gradients in a row")
                    continue
                skipped = 0
                optimizer.step()
                running += loss.item()
            # +1 so the first print averages a full window rather than one batch
            if (batch_i + 1) % s.print_freq == 0:
                print(f"[epoch {epoch + 1}/{s.num_epochs}, batch {batch_i + 1:5d}/{len(train_loader)}] "
                      f"loss: {running / (s.print_freq * steps_per_batch):.3f} lr: {scheduler.get_last_lr()[0]:.6f}")
                running = 0.0
        scheduler.step()

        # eval_every > 1 trades the per-epoch curve for wall clock; the last
        # epoch always evaluates so a run still ends with a real number
        if (epoch + 1) % s.eval_every != 0 and epoch != s.num_epochs - 1:
            continue
        model.eval()
        with torch.no_grad():
            value = evaluate(model, val_loader)

        # last.pt every evaluation so a crash costs little; best.pt only on
        # improvement, so it survives later overfitting
        save_checkpoint(s.checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, value, metric_name)
        if value > best:
            best = value
            save_checkpoint(s.checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, value, metric_name)
            print(f"new best {metric_name}: {value:.4f} (epoch {epoch + 1})")
        if on_epoch_end is not None:
            on_epoch_end(epoch, value)

    print(f"Training complete -- checkpoints in {s.checkpoint_dir}")
    return model
