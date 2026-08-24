# Project Tickets

## Phase / Milestone Map

Two levels of naming to keep straight:

**Phases** = resume-level story arc:
- Phase 1: the existing LiDAR SLAM repo (already done, uses pretrained models + GTSAM).
- Phase 2: this project (VIS-001 → VIS-031). Learned RGB descriptor + classical
  pose estimator, trained from scratch.
- Phase 3: the Transformer extension (VIS-032+). Scene-conditioned pose graph
  optimization.

**Milestones** = internal structure within a phase:
- Milestones 0-6 live inside Phase 2.
- Milestones 7-10 live inside Phase 3.

## Fast Path Notes (skip guidance)

Everything below is nice-to-have; the tickets marked below are the ones that
directly unlock Phase 3. Anything not on this list is optional / cuttable if
timeboxed:

- Milestone 1-2 (geometry + classical pipeline): all tickets needed. Do the
  math yourself, especially VIS-007 (Jacobian) and VIS-013 (Gauss-Newton on
  SE(3)) — this math directly reappears in Phase 3.
- Milestone 3 (descriptor training): all needed. Phase 3 consumes this.
- Milestone 4 (integration): VIS-019-020 needed. VIS-021 (confidence) and
  VIS-022 (comparison) are polish, not blockers.
- Milestone 5 (trajectory viz): all needed for the "visual" you asked for.
- Milestone 6 (uncertainty extension VIS-028-031): entirely skippable. Not
  related to Phase 3.

Total estimate to Phase 3 completion: ~4-5 months of focused work.

---



## Milestone 0 — Repository Setup

### VIS-001 — Create project skeleton
- Add `cpp/`, `learning/`, `tests/`, `data/`, and `experiments/`.
- Configure CMake, Eigen, and a C++ unit-test framework.
- Add Python environment and PyTorch MPS sanity check.

### VIS-002 — Define coordinate-frame conventions
- Document camera-frame axes.
- Define whether transforms map world-to-camera or camera-to-world.
- Define twist ordering and left/right perturbation convention.

## Milestone 1 — Geometry Foundation

### VIS-003 — Implement `SO3`
- `Exp`, `Log`, inverse, composition, hat/vee.
- Unit tests for round trips and small-angle behavior.

### VIS-004 — Implement `SE3`
- `Exp`, `Log`, inverse, composition, point transform.
- Implement left and right pose perturbations.

### VIS-005 — Implement the `SE3` adjoint
- Add `adjoint()`.
- Write tests showing conversion between left and right perturbations.

### VIS-006 — Implement pinhole camera model
- Projection and back-projection.
- Intrinsics handling.
- Numerical edge-case tests.

### VIS-007 — Derive and implement reprojection Jacobian
- Derive the Jacobian w.r.t. pose perturbation.
- Implement analytical form.
- Compare with central finite differences.

## Milestone 2 — Classical Relative Pose Baseline

### VIS-008 — Load a small RGB/depth/pose dataset subset
- Start with a handful of sequences.
- Build deterministic train/test utilities.

### VIS-009 — Generate geometric ground-truth correspondences
- Back-project pixels with depth.
- Transform points into the second camera.
- Reproject and reject occluded/invalid matches.

### VIS-010 — Implement normalized eight-point essential matrix solver
- Use Eigen SVD.
- Enforce essential-matrix rank constraints.

### VIS-011 — Implement RANSAC around essential-matrix estimation
- Fit, score, and refine hypotheses.
- Record inlier ratio and pose error.

### VIS-012 — Recover relative pose from the essential matrix
- Decompose candidate poses.
- Resolve ambiguity using cheirality.

### VIS-013 — Implement nonlinear pose refinement
- Minimize geometric/reprojection residuals with Gauss-Newton.
- Use the analytical Jacobians from VIS-007.

## Milestone 3 — Learned Descriptor Model

### VIS-014 — Build training sample generator
- Produce positive correspondence pairs from RGB/depth/pose.
- Produce spatial and hard-negative samples.

### VIS-015 — Implement first CNN descriptor network
- Residual blocks.
- Multi-scale feature map.
- 64- or 128-D descriptors.

### VIS-016 — Implement contrastive training objective
- Start with InfoNCE.
- Add descriptor normalization.
- Log positive/negative distance distributions.

### VIS-017 — Train first model on M1 Max
- Verify stable MPS training.
- Establish batch size, memory use, and throughput.
- Save reproducible checkpoints.

### VIS-018 — Evaluate descriptor quality
- Recall@1 / recall@5.
- Match precision/recall.
- Compare against at least one classical baseline.

### VIS-018b — Dilated convolutions for receptive field (LEARN FIRST)
Deferred until dilation is understood — do not let it be added blind.

Context: the stage-1 sweep found receptive field to be the dominant factor by a
wide margin. RF 15 -> 29 (kernel 3 -> 5) was +18.6pp MMA@8, and the four
architecture variants line up monotonically on RF alone:

    RF  9 (strides 1,2,2) -> 15.89%
    RF 15 (kernel 3)      -> 17.86%
    RF 23 (kernel 7,5,3)  -> 27.26%
    RF 29 (kernel 5)      -> 36.46%

Bigger kernels buy RF but cost parameters quadratically: kernel 9 reaches RF 57
for 3.4M params vs 0.4M at baseline. Dilation reaches the same distances at
kernel-3 cost, so it is the obvious next lever once the concept is understood.

To read first: what dilation does to the sampling grid, why the effective kernel
is (k-1)*d + 1, and why 'same' padding then needs dilation*(k-1)//2 rather than
k//2 (get that wrong and the feature map shrinks, silently breaking every
coordinate conversion downstream).

## Milestone 4 — Learned Geometry Integration

### VIS-019 — Export learned descriptors for C++ evaluation
- Define a simple interchange format or inference boundary.
- Keep Python training separate from C++ geometry.

### VIS-020 — Use learned matches in the C++ pose estimator
- Run learned correspondences through RANSAC + pose refinement.
- Measure relative pose error and inlier ratio.

### VIS-021 — Add learned match confidence
- Extend the network to predict a confidence score.
- Use confidence as a weight in nonlinear refinement.

### VIS-022 — Compare classical vs learned pipelines
Classical baselines use OpenCV — do NOT hand-implement SIFT or ORB, they are
pure engineering craft with no math/ML learning value.
- `cv2.SIFT_create()` — quality-reference classical baseline.
- `cv2.ORB_create()` — speed-reference classical baseline (used in ORB-SLAM).
- Hold everything constant except the descriptor: same keypoint budget, same
  matcher (nearest-neighbor + Lowe's ratio test), same downstream C++ pose
  estimator.

Evaluate on TartanAir test set bucketed by difficulty:
- easy pairs (small baseline, ≤5-frame gap);
- medium pairs (10-20 frame gap);
- hard pairs (30+ frame gap, corner turns);
- adversarial pairs (weak texture, motion blur, lighting change).

For each descriptor variant (SIFT, ORB, learned, learned+confidence) × each
difficulty bucket, report rotation error, translation error, inlier ratio,
runtime. Headline: learned should match SIFT on easy, beat SIFT on hard.

Optional stretch: add SuperPoint (pretrained) as a "modern learned ceiling"
comparison. Skip if timeboxed.

## Milestone 5 — Sequence Trajectories (Visual Odometry)

Goal: chain relative pose estimates across a video to produce a 3D trajectory.
No global optimization here — that lives in Phase 3. This milestone gives you
a real 3D visualization ("more than 2 images side by side") without needing a
factor graph solver yet.

### VIS-023 — Sequence loading and pair generation
- Extend the dataset loader to yield consecutive-frame pairs `(i, i+k)` for
  configurable `k` (frame stride).
- Preserve absolute ground-truth poses for evaluation.

### VIS-024 — Chain relative poses into a global trajectory
- For each consecutive pair, estimate `T_{i+1, i}` using the C++ pose estimator
  (both classical and learned variants).
- Anchor the trajectory at `T_W0 = I` and compose forward:
  `T_Wi = T_W(i-1) · T_(i-1, i)` (or equivalent inverse composition per your
  convention).
- Save trajectories to disk as N×4×4 arrays for both pipelines.

### VIS-025 — 3D trajectory visualization
- Plot ground-truth and estimated trajectories together in 3D (matplotlib
  `Axes3D` or Open3D).
- Include camera frustums at keyframes to show orientation, not just position.
- Save as static images + optionally an animation.

### VIS-026 — Per-frame drift metrics
- Absolute Trajectory Error (ATE): RMS translation error after Umeyama alignment.
- Relative Pose Error (RPE): drift per meter or per second.
- Plot error growth over trajectory length — this is where you *see* drift
  and motivate global optimization.

### VIS-027 — Side-by-side pipeline comparison viz
- One figure per sequence with:
  - ground-truth trajectory,
  - classical VO trajectory,
  - learned VO trajectory,
  - overlaid on the same axes with distinct colors.
- Deliverable: a compelling one-image summary of "classical vs learned VO
  drift over N frames."

## Milestone 6 — Research Extension: Uncertainty and Ablations

### VIS-028 — Predict correspondence uncertainty
- Predict `sigma` rather than only confidence.
- Train a heteroscedastic loss.
- Use uncertainty in weighted least squares.

### VIS-029 — Calibration experiment
- Test whether predicted uncertainty tracks actual geometric error.
- Produce reliability/calibration plots.

### VIS-030 — Ablation study
Test effects of:
- descriptor dimension;
- hard-negative mining;
- feature pyramid depth;
- confidence/uncertainty head;
- robust loss choice.

### VIS-031 — Write technical report / README
Document:
- method;
- derivations;
- implementation details;
- baselines;
- results;
- failure cases;
- next research questions.

---

# Phase 3 — Learned Pose Graph Optimization

Goal: replace the classical Levenberg-Marquardt solver (as used in GTSAM) with
a learned model that refines a noisy trajectory using loop closure constraints.
Compare against a from-scratch dense Gauss-Newton baseline and against GTSAM
(from prior work). End goal: a trained model that could plausibly be dropped
into the LiDAR SLAM pipeline as a real-time GTSAM alternative.

## Milestone 7 — Loop Closure and Classical Baselines

### VIS-032 — Loop closure detection using existing descriptors
- Reuse the learned descriptors from Milestone 3 (no new training).
- Global-pool per-frame descriptors into a single fingerprint vector.
- Detect candidate loop closures by nearest-neighbor search in fingerprint space
  above a similarity threshold, with a minimum frame-index gap to avoid trivial
  matches.
- Validate candidates geometrically: run the pose estimator on the candidate
  pair, accept if inlier ratio exceeds a threshold.

### VIS-033 — Pose graph data structure
- Represent a pose graph as: `nodes[i] = T_Wi` (absolute pose estimates) and
  `edges = [(i, j, T_ij_measured, information_matrix)]`.
- Support serialization to disk.
- Support both odometry edges (consecutive) and loop closure edges (non-adjacent).

### VIS-034 — Dense Gauss-Newton baseline solver in C++
- Implement multi-pose Gauss-Newton for pose graph optimization from scratch,
  reusing existing SO(3), SE(3), hat, adjoint code.
- Dense Jacobian and Hessian (fine for graphs <100 poses).
- Left perturbation on SE(3), per project convention.
- NOT GTSAM. This is the "hand-rolled classical baseline" — proves you
  understand what LM does and gives you a controlled comparison target.

### VIS-035 — Verify against GTSAM
- Install GTSAM (from prior LiDAR SLAM project) as a comparison-only dependency.
- On a handful of small synthetic pose graphs, verify that your dense GN solver
  and GTSAM converge to the same optimum within tolerance.
- If they disagree: your solver is wrong; do not proceed until they match.

## Milestone 8 — Synthetic Pose Graph Dataset

### VIS-036 — Synthetic trajectory generator
- Sample random smooth 3D trajectories (splines through random waypoints).
- Discretize into per-frame ground-truth absolute poses.
- Parameterize by trajectory length, curvature, and total distance.

### VIS-037 — Simulated frontend noise
- Compute clean relative poses between consecutive frames.
- Corrupt with configurable Gaussian noise on translation and rotation
  (represented in the SE(3) tangent space).
- Model realistic noise scales (calibrate against actual noise from your VO
  frontend in Phase 2).

### VIS-038 — Simulated loop closures
- Randomly select frame pairs `(i, j)` with `|i-j| > threshold` as loop closures.
- Add clean relative-pose measurements for those, plus noise.
- Also add a fraction of *wrong* loop closures (false positives) to test
  robustness.

### VIS-039 — Dataset pipeline
- Generate ~10^5 synthetic pose graphs for training.
- Split into train / val / test.
- Save as a batched tensor format friendly to PyTorch dataloaders.

## Milestone 9 — Scene-Conditioned Learned Pose Graph Optimizer

Core hypothesis: a Transformer that sees both the pose graph AND the visual
context of the whole trajectory can learn environment-specific priors ("rooms
are usually rectangular", "hallways are straight") that classical solvers
cannot encode. This is the main research contribution.

### VIS-040 — Loss functions on SE(3)
- Per-pose loss: L2 on translation, geodesic (`SO(3)::Log`-based) on rotation.
- Use the 6D rotation representation (Zhou et al. 2019) for the network's
  rotation output — avoids quaternion double-cover issues.
- Sum losses over all frames; weight translation vs rotation to comparable
  magnitudes.

### VIS-041 — Scene context aggregator
- Consume per-keyframe descriptors from the Phase 2 descriptor model (frozen).
- Aggregate into a single "scene vector" summarizing the whole trajectory seen
  so far.
- Start with mean pooling as a baseline.
- Upgrade to attention pooling: learn a query vector that pools descriptors
  adaptively into the scene vector. Model decides which frames matter most for
  scene characterization.
- Output: single fixed-size vector representing "what environment is this".

### VIS-042 — Scene-conditioned Transformer architecture
- Each keyframe is a token: token embedding = `[pose_features, descriptor]`.
- Prepend the scene vector from VIS-041 as a special `[SCENE]` token
  (BERT-`[CLS]`-style). Every pose token can attend to it in every layer.
- Odometry edges encoded as attention biases between adjacent tokens.
- Loop closure edges encoded as attention biases between non-adjacent tokens.
- K Transformer blocks refine the pose estimates.
- Output head: per-token 9-vector (3 for translation, 6 for rotation via 6D
  representation) applied as a delta on top of the input pose.

### VIS-043 — Training loop
- Train on synthetic pose graphs from Milestone 8, paired with cached
  descriptors from Phase 2 keyframes (or synthetic descriptors for pure
  synthetic graphs — see VIS-046).
- Validate on held-out graphs.
- Log convergence, per-epoch loss curves, and trajectory ATE / RPE.
- Freeze the descriptor CNN during Phase 3 training — only the pose Transformer
  and scene aggregator are learned here.

### VIS-044 — Robustness training
- Introduce false-positive loop closures during training.
- Add auxiliary loss for the model to flag suspicious edges.
- Alternative: train with a robust loss (Huber / Geman-McClure) rather than L2.

### VIS-045 — Diverse scene training corpus
- To learn scene priors, the training data must span diverse environment types.
- Use TartanAir sequences from *multiple* environments: indoor rooms, hallways,
  outdoor open spaces, forests. Not just one scene type.
- Track distribution of scene types in the training set — if imbalanced, weight
  the loss.

### VIS-046 — Pure-synthetic vs real-features training regime
- Pure-synthetic: generate fake descriptors alongside fake pose graphs. Model
  never sees real images. Trains fast, generalizes poorly.
- Real-features: use actual Phase 2 descriptors on TartanAir keyframes, paired
  with pose graphs derived from the ground truth. Slower, more realistic.
- Compare both — this ablation shows whether the model actually benefits from
  real scene context or just learns generic optimization.

## Milestone 10 — Evaluation and Deployment

### VIS-047 — Evaluation harness
Compare on synthetic and real pose graphs:
- No-optimization baseline (raw frontend output).
- Dense Gauss-Newton (from VIS-034).
- GTSAM (as ceiling reference).
- Learned Transformer WITHOUT scene context (pose-only ablation).
- Learned Transformer WITH scene context (main system).

Metrics: ATE, RPE, inference time, robustness to noise level, robustness to
false-loop-closure rate.

### VIS-048 — Scene context ablation — the money experiment
- Compare pose-only Transformer vs scene-conditioned Transformer on:
  - Indoor room sequences (where structural priors should help most).
  - Hallway/corridor sequences (long thin structure).
  - Outdoor open-space sequences (where priors should help least).
- Predicted result: scene context helps most on indoor/structured scenes,
  helps little on open outdoor scenes.
- If this pattern shows up cleanly, it *is* your headline result.

### VIS-049 — Attention visualization
- Extract attention maps between pose tokens and the scene token.
- Extract attention maps between loop-closure pose pairs.
- Show qualitatively what the model attends to when refining tough cases.
- Deliverable: figures for the writeup showing "here's what the model looked
  at when it decided to trust/reject this loop closure."

### VIS-050 — Scaling and generalization study
- Train on graphs of size N, evaluate on size 2N, 4N. Does positional encoding
  extrapolate?
- Train on noise level σ, evaluate on 2σ. Does it degrade gracefully?
- Ablate: number of Transformer layers, scene vector dimension, aggregation
  strategy.

### VIS-051 — Deploy to LiDAR SLAM pipeline (integration bonus)
- Wrap the trained optimizer in a ROS2 node that mirrors GTSAM's interface in
  the LiDAR SLAM repo.
- Swap in as a drop-in GTSAM replacement.
- Compare trajectory quality on iPhone LiDAR data collected in the prior
  project.
- Deliverable: real-time deployment on-device or on the ROS2 host.

### VIS-052 — Writeup for Phase 3
Document:
- motivation: classical optimizers can't encode scene priors;
- architecture: scene-conditioned Transformer;
- synthetic data generation and diverse-scene training corpus;
- training procedure;
- headline result: scene context helps on structured environments where
  classical priors would have to be hand-coded;
- attention visualization figures;
- deployment story (integration into LiDAR SLAM);
- failure cases (open outdoor scenes, out-of-distribution environments);
- open research questions.
