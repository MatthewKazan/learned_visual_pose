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
from dataclasses import asdict
from functools import partial

from data_utils.loop_closure_dataset import LoopClosureDataset
from data_utils.wide_baseline_CNN import WideBaselineDataset
from visual_pose.checkpoints import load_cnn_model
from visual_pose.config import Config
from visual_pose.data_utils.loaders import build_wide_baseline_loaders, build_loop_closure_loaders, \
    build_hm3d_wide_baseline_loaders
from visual_pose.matching import mma, test_model
from visual_pose.training import fit
from visual_pose.training_CNN import cnn_evaluate, cnn_step

# The run whose best.pt seeds this one. Architecture fields in Config must match
# its config.json or load_state_dict fails -- verified identical as of writing.
PRETRAINED_RUN = "fine_tuning_wide_baseline_2"


def main():
    cfg = Config()
    # A fine-tune, not a retrain: 10x lower LR and a short schedule, so the
    # model adapts to the new pairs instead of relearning from them.
    cfg.learning_rate = 0.001
    cfg.num_epochs = 3

    # load from the pretrained run, save to a new one. checkpoint_dir is derived
    # from run_name, so the order matters -- reading before the rename would
    # look for best.pt inside a directory that does not exist yet.
    cfg.run_name = PRETRAINED_RUN
    model = load_cnn_model(cfg)
    print(f"initialised from {cfg.checkpoint_dir / 'best.pt'}")
    cfg.run_name = "fine_tuning_wide_baseline_2"

    train_loader, val_wide_loader, val_narrow_loader = build_hm3d_wide_baseline_loaders(cfg, train_trajectories=100, val_trajectories=20)

    # Checkpoints on the wide metric. Narrow is the tripwire, not a target:
    # 5-frame pairs are what the odometry actually runs on.
    def report_narrow(epoch_i, val_wide_mma):
        narrow_mma = mma(test_model(model, val_narrow_loader), cfg.checkpoint_tau).item()
        print(f"  epoch {epoch_i + 1}: wide MMA@{cfg.checkpoint_tau} {val_wide_mma:.2%}"
              f" | narrow MMA@{cfg.checkpoint_tau} {narrow_mma:.2%}")

    baseline = mma(test_model(model, val_narrow_loader), cfg.checkpoint_tau).item()
    print(f"before fine-tuning: narrow MMA@{cfg.checkpoint_tau} {baseline:.2%}")

    # fit builds a fresh optimizer and scheduler, on purpose: the saved ones are
    # 40 epochs into a cosine decay with momentum fitted to the old distribution.
    fit(model, train_loader, val_wide_loader,
        step=partial(cnn_step, temperature=cfg.temperature),
        evaluate=partial(cnn_evaluate, tau=cfg.checkpoint_tau),
        settings=cfg.train_settings(), metric_name=f"wide MMA@{cfg.checkpoint_tau}",
        config=asdict(cfg), on_epoch_end=report_narrow)
    print("Next: rerun experiments/cpp_testing.py and check whether loop closure "
          "inlier counts rose -- that is the result this is for, not MMA.")


if __name__ == "__main__":
    main()
