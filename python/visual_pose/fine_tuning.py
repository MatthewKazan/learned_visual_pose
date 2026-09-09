"""
Fine-tune the descriptor on wide-viewpoint pairs.

Why: measured on P000, the descriptor returns ZERO correct matches on revisit
pairs (350-frame gap, ~30 deg heading change) while returning 25-82% correct on
consecutive pairs. RANSAC then finds a 25-116 point consensus among mutually
consistent WRONG matches, which no inlier-count threshold can detect -- so loop
closure fails and the pose graph has nothing sound to optimise. The cause is a
train/deploy mismatch: `frame_gap = (2, 5, 10)` means the model has never seen
a pair like that.
"""
import json

from visual_pose.checkpoints import load_model
from visual_pose.config import Config
from visual_pose.data_utils.loaders import build_wide_baseline_loaders
from visual_pose.matching import mma, test_model
from visual_pose.training import set_up_loss_optimizer_lr_scheduler, train_val_model

# The run whose best.pt seeds this one. Architecture fields in Config must match
# its config.json or load_state_dict fails -- verified identical as of writing.
PRETRAINED_RUN = "k7_d122_mma70"


def main():
    cfg = Config()
    # A fine-tune, not a retrain: 10x lower LR and a short schedule, so the
    # model adapts to the new pairs instead of relearning from them.
    cfg.learning_rate = 0.001
    cfg.num_epochs = 5

    # load from the pretrained run, save to a new one. checkpoint_dir is derived
    # from run_name, so the order matters -- reading before the rename would
    # look for best.pt inside a directory that does not exist yet.
    cfg.run_name = "fine_tuning_wide_baseline"
    model = load_model(cfg)
    print(f"initialised from {cfg.checkpoint_dir / 'best.pt'}")
    cfg.run_name = "fine_tuning_wide_baseline"

    train_loader, val_wide_loader, val_narrow_loader = build_wide_baseline_loaders(cfg)

    # Fresh optimizer and scheduler on purpose. The saved ones are 40 epochs
    # into a cosine decay with momentum buffers fitted to the old distribution.
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

    # Checkpoints on the wide metric. Narrow is the tripwire, not a target:
    # 5-frame pairs are what the odometry actually runs on.
    def report_narrow(epoch_i, val_wide_mma):
        narrow_mma = mma(test_model(model, val_narrow_loader), cfg.checkpoint_tau).item()
        print(f"  epoch {epoch_i + 1}: wide MMA@{cfg.checkpoint_tau} {val_wide_mma:.2%}"
              f" | narrow MMA@{cfg.checkpoint_tau} {narrow_mma:.2%}")

    baseline = mma(test_model(model, val_narrow_loader), cfg.checkpoint_tau).item()
    print(f"before fine-tuning: narrow MMA@{cfg.checkpoint_tau} {baseline:.2%}")

    train_val_model(model, train_loader, val_wide_loader, loss_fn, optimizer,
                    lr_scheduler, num_epochs=cfg.num_epochs,
                    print_freq=cfg.print_freq, checkpoint_dir=cfg.checkpoint_dir,
                    checkpoint_tau=cfg.checkpoint_tau, on_epoch_end=report_narrow)
    print(f"Fine-tuning complete -- checkpoints in {cfg.checkpoint_dir}")
    print("Next: rerun experiments/cpp_testing.py and check whether loop closure "
          "inlier counts rose -- that is the result this is for, not MMA.")


if __name__ == "__main__":
    main()
