"""
Descriptor training entrypoint. Every tunable lives in visual_pose.config.Config.

    python -m visual_pose.train
    python -m visual_pose.train --temperature 0.03 --norm group --run_name t03_gn
    python -m visual_pose.train --num_epochs 6 --train_sequences P000 P002 --run_name quick
"""
import argparse
import json
from pathlib import Path

import torch

from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE
from visual_pose.data_utils.loaders import build_loaders
from visual_pose.models.descriptor_cnn import DescriptorCNN
from visual_pose.training import train_val_model, set_up_loss_optimizer_lr_scheduler


def main(cfg: Config | None = None):
    if cfg is None:
        # --config-json lets an orchestrator hand over a config (e.g. a sweep
        # winner) without reinstating a flag per field. Anything absent from the
        # file keeps its Config default.
        ap = argparse.ArgumentParser()
        ap.add_argument("--config-json", default=None,
                        help="JSON dict of Config overrides")
        args = ap.parse_args()
        overrides = json.loads(Path(args.config_json).read_text()) if args.config_json else {}
        # JSON has no tuples, so tuple-typed fields come back as lists
        overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        cfg = Config(**overrides)

    print(f"run: {cfg.run_name}\n{cfg.summary()}\n")

    # seeds torch's RNG, which the DataLoader derives per-worker seeds from --
    # so grid offsets and correspondence subsampling replay too
    torch.manual_seed(cfg.seed)

    train_loader, val_loader = build_loaders(cfg)
    print(f"train {len(train_loader.dataset)} pairs / {len(train_loader)} batches, "
          f"val {len(val_loader.dataset)} pairs\n")

    model = DescriptorCNN.from_config(cfg).to(DEVICE)

    loss_fn, optimizer, lr_scheduler = set_up_loss_optimizer_lr_scheduler(
        model=model,
        learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,
        num_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        min_lr_factor=cfg.min_lr_factor,
        optimizer=cfg.optimizer,
        temperature=cfg.temperature,
    )

    # write the config beside the checkpoints so a result is always traceable
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (cfg.checkpoint_dir / "config.json").write_text(
        json.dumps(cfg.__dict__, indent=2, default=list))

    train_val_model(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler,
                    num_epochs=cfg.num_epochs, print_freq=cfg.print_freq,
                    checkpoint_dir=cfg.checkpoint_dir, checkpoint_tau=cfg.checkpoint_tau)
    print(f"Training complete -- checkpoints in {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
