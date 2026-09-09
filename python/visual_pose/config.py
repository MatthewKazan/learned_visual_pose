"""
Every tunable in one dataclass, with CLI flags generated from its fields.

    python -m visual_pose.train --temperature 0.03 --norm group --run_name t03_gn

Written into each checkpoint, so a result is always traceable to what produced
it, and a sweep is a command line rather than an edit.
"""
from dataclasses import dataclass, fields, asdict
from pathlib import Path

from visual_pose.data_utils.constants import REPO_DIR


@dataclass
class Config:
    # ---- data ----
    train_sequences: tuple = ("P000", "P001", "P002", "P004", "P005", "P006")
    val_sequence: str = "P000"
    # train spans several baselines; val is fixed so runs stay comparable
    frame_gap: tuple = (2, 5, 10)
    val_frame_gap: int = 5

    # fine tuning params
    min_frame_gap: int = 60
    overlap_ratio_band: tuple = (0.1, 0.6)
    wide_to_narrow_ratio: float = 2.0
    wide_val_pairs: int = 2000
    ##
    num_correspondences: int = 512   # 256/1024 both flat
    sample_step: int = 8
    # 0.0 (mask off) cost -0.76pp, so the dark-pixel filter earns its place
    darkness_threshold: float = 0.05
    max_depth: float = 30.0
    occlusion_tol: float = 0.1 # FROZEN, changes very little
    jitter: float = 0.2

    # ---- model ----
    # ~50 sweep trials on one proxy. Receptive field dominated everything else
    # by an order of magnitude, then saturated:
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
    # RF 79-131 is one noise band; past ~137 it degrades, since wide context is
    # view-dependent. Capacity saturates near 2M, so k7 ties k9 and k11. wide k9
    # was the only live result and cost 2.24x params for +2.36pp.
    body_channels: tuple = (64, 128, 256)
    body_kernel_sizes: tuple = (7, 7, 7)
    body_strides: tuple = (2, 2, 2)
    # Reach at fixed parameters, but only while the taps stay dense. k7
    # d(1,2,2) covers 100% of its RF box, k3 d(1,3,9) 9.6%, and at matched
    # RF ~86 that was 48.65% vs 28.42%. Coverage matters as much as reach.
    body_dilations: tuple = (1, 2, 2)
    descriptor_dim: int = 128
    norm: str = "batch"          # batch | group | none

    # ---- loss ----
    # 0.02 beat 0.07 by +1.08pp at RF 15, indistinguishable by RF 29. Kept
    # because every later result was measured there.
    temperature: float = 0.02

    # ---- optimization ----
    # adamw lost 1.5pp, but at lr=0.01 -- ~30x too high for it. That measured
    # a bad learning rate, not the optimiser.
    optimizer: str = "sgd"       # sgd | adamw
    learning_rate: float = 0.01  # 0.003 flat, 0.03 clearly worse (-1.4pp)
    momentum: float = 0.9
    # UNTESTED. Regularizers cost accuracy early and pay late, so a 6-epoch
    # proxy trial always says turn them down. Needs a full-length A/B.
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

    # ---- evaluation: the frontend / pose-graph run (visual_pose.app.run) ----
    max_edges: int | None = None      # None = the whole sequence
    # per-cell descriptor cosine for matching. On P001 closures 0.70 -> 0.80
    # halves the error (0.62 -> 0.28 deg) at 3.4x fewer matches; past 0.85
    # RANSAC finds no consensus.
    similarity_threshold: float = 0.7

    # metres a closure may disagree with the odometry chain. Measured: true
    # closures disagree 0.06-0.22 m, false ones 4.91-5.00 m.
    loop_max_dist: float = 2.0
    loop_min_gap: int = 10            # keyframes apart to count as a loop
    # absolute count, not the ratio: wrong closures had 3-8 inliers and right
    # ones 28-163, while the ratio fails both ways (real closures sit at
    # 0.04-0.12, a 3-of-5 fit reports 0.60 and is 179 deg wrong).
    loop_min_inliers: int = 25
    # fingerprint cosine deciding which pairs are proposed. 99% precision at
    # 18.4% recall on P003. Does not transfer: on P000 no threshold works,
    # since the true revisits score below the false ones.
    loop_retrieval_similarity: float = 0.94
    loop_max_candidates: int | None = None   # None = no cap

    weight_edges: bool = True         # information-weight the graph edges
    huber: float = 0.0                # robust kernel threshold; 0 = off
    note: str = ""                    # free tag, shown on the plot label

    inlier_threshold: float = 0.05
    degeneracy_threshold: float = 0.001

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
