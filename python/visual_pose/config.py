"""
Every tunable in one dataclass, with CLI flags generated from its fields.

    python -m visual_pose.train --temperature 0.03 --norm group --run_name t03_gn

Written into each checkpoint, so a result is always traceable to what produced
it, and a sweep is a command line rather than an edit.
"""
from dataclasses import dataclass, fields, asdict, replace
from pathlib import Path

from visual_pose.data_utils.constants import REPO_DIR


@dataclass
class TrainSettings:
    """What training.fit needs, whatever the model. Config.train_settings() builds one for the CNN."""
    num_epochs: int = 40
    optimizer: str = "sgd"       # sgd | adamw
    learning_rate: float = 0.01
    momentum: float = 0.9        # sgd only
    weight_decay: float = 1e-4
    min_lr_factor: float = 0.01  # cosine floor, as a fraction of learning_rate
    print_freq: int = 50         # batches between loss prints
    eval_every: int = 1          # epochs between evaluations; the last always runs
    run_name: str = "default"

    @property
    def checkpoint_dir(self) -> Path:
        return REPO_DIR / "checkpoints" / self.run_name


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
    similarity_threshold: float = 0.8
    odometry_max_rotation_rad = 1.5
    # Refuse a closure whose inlier residuals are near-planar: largest/smallest
    # eigenvalue of the point covariance. Inverting one claimed 0.001 mm along
    # the plane normal, and that single edge carried 99% of the graph's cost
    # (00820, 00875). 10 HM3D graphs, 2026-10-02: odometry peaks at 9.4e4; 1e6
    # refuses 3.1% of closures, including 00875 #447 (1.6e10) and 00820 #429
    # (8.4e8). Closures only -- refusing an odometry edge forks the chain.
    info_ratio_threshold: float = 1e6
    # Refuse a closure before the d^2 test when its rotation disagrees with the
    # chain by more than this (rad; 1.5 = 86 deg). Across unknown-covariance
    # odometry the chain covariance is so wide that d^2 passed 94, 96 and 180 deg
    # closures on 00820 (d^2 9.5-92 < 300). Blind to translation-direction
    # errors: 00860's 90 deg one has a small rotation.
    max_residual_threshold: float = 1.5
    # Refuse a closure whose measured translation is under closure_collapse_ratio
    # of the chain's prediction, when the chain spans more than closure_collapse_min_m.
    # The 90 deg "direction" error on 00860 is this: 0.28 m measured against
    # 7.65 m true. Direction against the chain cannot separate (good closures
    # span 22 deg p50 to 177 deg, from drift). On the 10 saved HM3D graphs
    # (2026-10-02) this refuses 4 of 10 bad closures and 0 good ones the
    # rotation gate does not already refuse (161 good in 00820 are refused by both).
    closure_collapse_min_m: float = 2.0
    closure_collapse_ratio: float = 0.2
    # Lower bound on each principal sigma of an edge's inlier residuals (m).
    # True pairs sit at 0.14-1.7 mm (p5), the near-planar failures at
    # 0.001-0.007 mm; 0.1 mm leaves the first untouched and caps the second.
    point_sigma_floor_m: float = 1e-4

    loop_min_gap: int = 50            # keyframes apart to count as a loop
    # absolute count, not the ratio: wrong closures had 3-8 inliers and right
    # ones 28-163, while the ratio fails both ways (real closures sit at
    # 0.04-0.12, a 3-of-5 fit reports 0.60 and is 179 deg wrong).
    loop_min_inliers: int = 25
    # fingerprint cosine deciding which pairs are proposed. Currently INERT:
    # under backbone GeM pooling every eligible pair scores 0.97-0.99, so 0.94
    # and 0.98 give byte-identical runs and precision equals the 5.2% base
    # rate. Geometric verification is doing all the discriminating.
    loop_retrieval_similarity: float = 0.94
    # U-curve on P006: 25 -> 0.291, 200 -> 0.262, 500 -> 0.257, 3000 -> 0.303 m
    # ATE. Above the optimum the extra closures are still accurate, but they
    # stack on the same revisit event and their information sums as though
    # independent, so the graph over-trusts the loop.
    loop_max_candidates: int | None = 500
    # How the cap is spent. "topk" takes the highest cosines, which on P006
    # collapsed to a 16-keyframe span band (153-169) because the most similar
    # pairs cluster at the centre of the strongest overlap. "random" spreads
    # but discards the ranking; "span" keeps it inside span bands.
    # P006 ATE: topk 0.268, random 0.266, span 0.257.
    loop_selection: str = "span"      # topk | random | span
    loop_span_bands: int = 8          # bands for loop_selection="span"
    # Closure gate: Mahalanobis d^2 of a closure against the odometry chain's
    # accumulated covariance (WorldModel.chain_consistent_gate). Replaced a
    # 2 m disagreement limit, which encoded TartanAir's drift and rejected
    # every closure spanning a bad edge on HM3D. Not the chi^2(6) quantile
    # (12.6): chain drift is correlated, so the random-walk covariance runs
    # 4-8x under actual drift on TartanAir. Measured 2026-09-29 on P000, P003,
    # P006 and HM3D trajectory_20: true closures d^2 < 75, false ones > 3800.
    closure_gate_threshold: float = 300.0
    # Dense verification of a closure: fraction of frame i's pixels that, pushed
    # through its depth and the fitted transform, land on frame j's surface
    # within closure_depth_tolerance of their predicted depth. A degenerate fit
    # explains only the patch it matched: the wall closure on 00801
    # trajectory_23 (rotation right, translation 8 m off, 33 inliers, d^2 5.3
    # under the chain gate) scored 0.33 while 156 true closures scored 0.55-0.96.
    # Under ground truth true pairs score 0.92 (HM3D) and 0.54 (P000, occlusion
    # at wide baseline); shifted 0.5 m, under 0.1. Relative tolerance because
    # depth precision scales with depth on both datasets. Cannot see a wrong
    # translation on far-only geometry (P000's false closures score 0.68); the
    # chain gate covers those. Measured 2026-09-29.
    closure_depth_tolerance: float = 0.02
    closure_min_agreement: float = 0.35

    weight_edges: bool = True         # information-weight the graph edges
    # edge_information sums one term per inlier, which is right only if point
    # errors are independent. Against ground truth the Mahalanobis d^2 of an
    # odometry edge grew with its inlier count, d^2/N = 0.10-0.21 on both
    # datasets (chi^2(6) median 5.35 wants 0.003 at N=1800), so the errors are
    # shared -- depth bias, descriptor localisation -- and an edge is worth
    # about this many independent points however many it matched.
    edge_effective_inliers: int = 40
    # An odometry edge below this many inliers is kept (refusing it forks the
    # chain) but with an "unknown" covariance instead of one from its own
    # residuals: 5 inliers on a blank wall gave a 103 deg edge with a
    # centimetre-level covariance (HM3D trajectory_20, frames 336-344), and
    # the closure gate can only route around a bad edge whose covariance
    # admits it. Good odometry edges have ~1800 inliers; the bad ones had 5 and 21.
    # 100 declared good 93-inlier edges unknown and cut the anchor loose; 50 over
    # 24 HM3D trajectories: mean graph ATE 0.37 -> 0.20 m, trajectory_08 3.12 ->
    # 0.33 m, one sequence worse by 0.06 m (2026-09-29).
    odometry_min_inliers: int = 50
    unknown_edge_sigma_t: float = 1.0    # metres
    unknown_edge_sigma_r: float = 0.52   # radians, ~30 deg
    # Inert on every sequence measured (identical ATE at 0, 2, 5): there are no
    # outlier closures for it to suppress. Kept as insurance, not as a fix --
    # the failure mode that does bite is correlated closures, which a robust
    # kernel cannot see.
    huber: float = 2.0
    note: str = ""                    # free tag, shown on the plot label

    inlier_threshold: float = 0.05
    degeneracy_threshold: float = 0.001

    lm_lambda_init: float = 1.0

    # ---- loop-closure aggregator (train.train_atn_pooling) ----
    # The one definition of a true loop closure, for training and evaluation
    # alike: either frame sees more than this fraction of its sample grid in
    # the other. One threshold everywhere: training on 0.3 / 0.05 with the band
    # dropped, then scoring at 0.1, never showed the model the 0.1-0.3 pairs it
    # was scored on. 0.25, not 0.1: closer to what verification can match. Not
    # comparable with runs before 2026-10-02 (GeM span precision on 00800:
    # 51.8% at 0.1, 39.2% at 0.25).
    loop_true_overlap: float = 0.15
    aggregator_frozen_cnn: str = "fine_tuning_wide_baseline_2"   # pooled, and the frontend's matcher; must equal run.py BASE.run_name
    aggregator_run_name: str = "attention_pooling_infonce_t02_h8_ft2_e16"   # the rest in checkpoints/: earlier runs, kept to compare
    # AdamW, the usual choice for attention layers, not the CNN's SGD 0.01. Untuned.
    aggregator_optimizer: str = "adamw"
    aggregator_learning_rate: float = 1e-4
    aggregator_num_epochs: int = 16   # val peaked at epoch 11 (63.6%), flat after
    aggregator_anchors: int = 32          # InfoNCE rows per trajectory step
    aggregator_hard_negatives: int = 8    # per anchor: its own top-scoring false partners
    aggregator_temperature: float = 0.02  # with 8 hard negatives: val 63.6% vs GeM 53.4%; 0.07 / 2 plateaued near GeM
    # Train on every 4th keyframe from a random offset: neighbours are 0.6 m /
    # 13.5 deg apart and see nearly the same surfaces, and the CNN pass over a
    # trajectory's frames is most of a step (~11 ms a frame). ~60 frames a step
    # at 4, so ~10 min an epoch over 800 trajectories. Val keeps every frame.
    aggregator_frame_stride: int = 4
    # Trajectories per epoch, a fresh random draw each epoch: 200 of the 800
    # x ~60 frames is ~12k CNN passes, ~3 min, and every scene recurs within a
    # few epochs.
    aggregator_trajectories_per_epoch: int = 200
    # Optimizer steps per trajectory visit, on its cached CNN maps. 1, not 4:
    # on 24 trajectories x 400 updates, 4 draws stalled the loss (3.64 -> 3.69)
    # and ended val span 34.8% against 44.3% with 1 (GeM 40.5%), 2026-10-02.
    aggregator_draws_per_trajectory: int = 1
    # "sigmoid": every pair against one learned threshold (aggregator_sigmoid_step);
    # "infonce": per-anchor rows with self-mined hard negatives (aggregator_step).
    aggregator_loss: str = "infonce"   # sigmoid: 53.0% val, 9 points below
    # ColorJitter on training frames. Off: the CNN is frozen, so jitter shifts
    # its features in training only; same test, val span 38.6% with it, 44.3% without.
    aggregator_augment: bool = False
    aggregator_train_probe_scenes: int = 10   # trained-on trajectories scored beside val, to spot memorising

    # Closures whose both endpoints lie within this many keyframes of a
    # cluster's first closure are one revisit and share one closure's weight
    # (WorldModel.calibrate). UNTUNED.
    loop_closure_range: int = 5

    @property
    def checkpoint_dir(self) -> Path:
        # per-run directory so runs stop overwriting each other's best.pt
        return REPO_DIR / "checkpoints" / self.run_name

    def summary(self) -> str:
        return "\n".join(f"  {k:22} {v}" for k, v in asdict(self).items())

    def train_settings(self) -> TrainSettings:
        return TrainSettings(**{f.name: getattr(self, f.name) for f in fields(TrainSettings)
                                if hasattr(self, f.name)})

    def aggregator_train_settings(self) -> TrainSettings:
        return replace(self.train_settings(), run_name=self.aggregator_run_name,
                       optimizer=self.aggregator_optimizer,
                       learning_rate=self.aggregator_learning_rate,
                       num_epochs=self.aggregator_num_epochs)

    def common(self) -> dict:
        return dict(num_correspondences=self.num_correspondences,
                    sample_step=self.sample_step,
                    darkness_threshold=self.darkness_threshold,
                    max_depth=self.max_depth,
                    occlusion_tol=self.occlusion_tol)
