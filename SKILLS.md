# Skills to Focus On

## 1. Multi-View Geometry
Refresh and become comfortable deriving:
- pinhole camera model and homogeneous coordinates;
- intrinsic/extrinsic calibration;
- epipolar geometry;
- fundamental and essential matrices;
- triangulation;
- PnP;
- reprojection error;
- bundle adjustment.

Target: implement the important pieces yourself rather than relying entirely on OpenCV solvers.

## 2. Lie Groups and Manifold Optimization
Focus heavily on:
- `SO(3)` and `SE(3)`;
- `Exp` / `Log` maps;
- tangent spaces and perturbations;
- left vs. right perturbations;
- adjoint representation;
- analytical Jacobians;
- numerical Jacobian checking;
- Gauss-Newton and Levenberg-Marquardt;
- robust losses.

Target: derive the reprojection Jacobian with respect to an `SE(3)` pose perturbation and verify it against finite differences.

## 3. Modern C++ for Perception
Practice:
- C++17/20;
- Eigen;
- value/reference semantics and lifetimes;
- templates where useful;
- clean numerical APIs;
- unit testing;
- profiling and performance-aware data structures;
- CMake.

Target: keep geometry and optimization code independent of ML code.

## 4. Deep Learning for Correspondence
Learn through implementation:
- convolutional feature encoders;
- residual networks;
- feature pyramids;
- local descriptors;
- metric learning;
- InfoNCE / contrastive losses;
- hard-negative mining;
- augmentation;
- confidence and uncertainty prediction;
- evaluation of learned representations.

Target: design and train the model yourself rather than wrapping a pretrained matcher.

## 5. PyTorch Training Practice
Focus on:
- custom `Dataset` / `DataLoader` code;
- training loops;
- mixed precision where supported;
- checkpointing;
- train/validation splits;
- experiment tracking;
- debugging overfitting and unstable losses;
- profiling on Apple MPS.

## 6. Research Skills
For every major design decision:
- define a baseline;
- define a metric;
- run an ablation;
- record failure cases;
- separate training, validation, and test scenes;
- document what changed and why.

Useful initial metrics:
- descriptor recall@k;
- correspondence precision/recall;
- inlier ratio;
- relative rotation error;
- relative translation-direction error;
- runtime per image pair.
