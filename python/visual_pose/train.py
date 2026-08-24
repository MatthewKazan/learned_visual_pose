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
from torch.utils.data import DataLoader, ConcatDataset

from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.data_utils.training_dataset import TACorrespondenceDataset
from visual_pose.models.descriptor_cnn import DescriptorCNN
from visual_pose.models.training import train_val_model, set_up_loss_optimizer_lr_scheduler


def build_loaders(cfg: Config):
    root = REPO_DIR / "data" / "tartan_air"
    common = dict(num_correspondences=cfg.num_correspondences,
                  sample_step=cfg.sample_step,
                  darkness_threshold=cfg.darkness_threshold,
                  max_depth=cfg.max_depth,
                  occlusion_tol=cfg.occlusion_tol)

    # augment on train only -- val has to stay a fixed yardstick
    train_dataset = ConcatDataset([
        TACorrespondenceDataset(TartanAirSequence(root / name),
                                frame_gap=list(cfg.frame_gap), augment=True,
                                jitter=cfg.jitter, **common)
        for name in cfg.train_sequences
    ])
    val_dataset = TACorrespondenceDataset(TartanAirSequence(root / cfg.val_sequence),
                                          frame_gap=cfg.val_frame_gap, eval=True, **common)

    # persistent_workers keeps workers alive across epochs. On macOS they are
    # spawned, not forked, so otherwise every epoch re-imports torch in each one.
    loader = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0,
                  pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(train_dataset, shuffle=True, **loader),
            DataLoader(val_dataset, shuffle=False, **loader))


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

    model = DescriptorCNN(
        body_channels=list(cfg.body_channels),
        body_kernel_sizes=list(cfg.body_kernel_sizes),
        body_strides=list(cfg.body_strides),
        body_dilations=list(cfg.body_dilations),
        descriptor_dim=cfg.descriptor_dim,
        norm=cfg.norm,
    ).to(DEVICE)

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
