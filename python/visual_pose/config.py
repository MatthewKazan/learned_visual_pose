"""
One dataclass holding every tunable, plus CLI overrides generated from its fields.

    from visual_pose.config import Config
    cfg = Config.from_cli()          # defaults, overridden by any --flag given

    python -m visual_pose.train --temperature 0.03 --norm group --run_name t03_gn

Two reasons this beats constants in train.py: sweeps become command lines instead
of edits, and the whole config can be written into the checkpoint, so a result is
always traceable to what produced it.
"""
from dataclasses import dataclass, fields, asdict
from pathlib import Path

from visual_pose.data_utils.constants import REPO_DIR


@dataclass
class Config:
    # ---- data ----
    train_sequences: tuple = ("P000", "P001", "P002", "P004", "P005", "P006")
    val_sequence: str = "P003"
    # train spans several baselines; val stays fixed at 5 so the metric is
    # comparable across runs (gap-2 pairs are much easier than gap-10)
    frame_gap: tuple = (2, 5, 10)
    val_frame_gap: int = 5
    num_correspondences: int = 512   # 256/1024 both flat
    # 4 and 12 lost badly, but that was an eval artifact, not training: with
    # eval=True the grid starts at (0,0), so a step that is not a multiple of the
    # network stride puts half the queries between feature cells and they get
    # interpolated. 8 and 16 are both aligned; 16 was +1.7pp, inside the floor.
    sample_step: int = 8
    # 0.0 (mask off) cost -0.76pp, so the dark-pixel filter earns its place
    darkness_threshold: float = 0.05
    max_depth: float = 30.0
    occlusion_tol: float = 0.1 # FROZEN, changes very little
    jitter: float = 0.2

    # ---- model ----
    # From ~50 sweep trials, all on the same proxy so the numbers compare
    # directly. Receptive field dominated everything else by an order of
    # magnitude, then saturated:
    #
    #     params   RF  config              MMA@8
    #    1.06M     29  k5 d(1,1,1)         36.46%
    #    1.06M     53  k5 d(1,2,2)         51.70%
    #    2.05M     43  k7 d(1,1,1)         48.13%
    #    2.05M     79  k7 d(1,2,2)         59.22%  <-- chosen
    #    2.05M    103  k7 d(1,2,3)         57.08%
    #    3.37M     89  k9 d(1,1,2)         59.59%
    #    3.37M    105  k9 d(1,2,2)         57.82%
    #    3.37M    169  k9 d(1,2,4)         51.99%
    #    5.01M    131  k11 d(1,2,2)        59.17%
    #    6.99M    157  k13 d(1,2,2)        56.02%
    #    7.54M    105  wide k9 d(1,2,2)    60.18%
    #
    # RF 79-131 is one noise band; past ~137 it degrades (too much context is
    # view-dependent, so a point stops matching itself across a baseline).
    # Capacity saturates around 2M -- k7 ties k9 and k11. wide k9 was the only
    # statistically live result and cost 2.24x params for +2.36pp, so k7 is the
    # cost-adjusted pick.
    body_channels: tuple = (64, 128, 256)
    body_kernel_sizes: tuple = (7, 7, 7)
    body_strides: tuple = (2, 2, 2)
    # Dilation buys reach at fixed parameters, but only while the kernel has
    # enough taps to stay dense. Measured coverage (fraction of the RF box that
    # actually reaches the output): k7 d(1,2,2) is 100%, whereas k3 d(1,3,9) is
    # 9.6% -- and at matched RF ~86, 100% coverage scored 48.65% against 28.42%.
    # Coverage is worth about as much as reach; do not raise these blindly.
    body_dilations: tuple = (1, 2, 2)
    descriptor_dim: int = 128
    norm: str = "batch"          # batch | group | none

    # ---- loss ----
    # 0.02 beat 0.07 by +1.08pp at RF 15 but the two were indistinguishable by
    # RF 29 -- the benefit was receptive-field dependent. Kept at 0.02 since all
    # stage 2-4 results were measured there.
    temperature: float = 0.02

    # ---- optimisation ----
    # adamw lost by 1.5pp, but at lr=0.01 -- roughly 30x too high for it. That
    # trial measured a bad learning rate, not the optimiser; it needs its own
    # sweep before being ruled out.
    optimizer: str = "sgd"       # sgd | adamw
    learning_rate: float = 0.01  # 0.003 flat, 0.03 clearly worse (-1.4pp)
    momentum: float = 0.9
    # STILL UNTESTED. Regularizers cost accuracy early and pay late, so every
    # 6-epoch proxy trial would have said turn them down. Needs a full-length
    # A/B -- and matters more now at 2M params than it did at 0.4M.
    weight_decay: float = 1e-4
    min_lr_factor: float = 0.01

    # ---- run ----
    batch_size: int = 8
    num_epochs: int = 40
    num_workers: int = 4
    seed: int = 0
    run_name: str = "default"
    checkpoint_tau: int = 8
    print_freq: int = 50

    @property
    def checkpoint_dir(self) -> Path:
        # per-run directory so runs stop overwriting each other's best.pt
        return REPO_DIR / "checkpoints" / self.run_name

    def summary(self) -> str:
        return "\n".join(f"  {k:22} {v}" for k, v in asdict(self).items())

    def common(self) -> dict:
        return dict(num_correspondences=self.num_correspondences,
                    sample_step=self.sample_step,
                    darkness_threshold=self.darkness_threshold,
                    max_depth=self.max_depth,
                    occlusion_tol=self.occlusion_tol)
