# Learned Visual Correspondence + Relative Pose Estimation

## Goal
Build and train an image feature model that finds reliable correspondences between two camera frames, then estimate relative camera pose with a mostly C++ geometric backend.

## Core Pipeline
1. Input two RGB images.
2. Run a custom PyTorch encoder to produce local descriptors and match confidence.
3. Match features between frames.
4. Estimate relative pose from correspondences.
5. Refine pose with nonlinear optimization on `SE(3)`.
6. Optionally extend to multi-view tracks and bundle adjustment.

## Intended Stack
- **Python / PyTorch:** model, training, dataset generation, experiments.
- **C++ / Eigen:** camera geometry, `SO(3)` / `SE(3)`, RANSAC, triangulation, pose refinement, optimization, evaluation.
- **OpenCV:** image I/O and basic preprocessing only where convenient.

## Dataset
Start with a dataset that provides RGB, depth, and ground-truth camera poses, such as TartanAir. Use depth + pose to generate ground-truth pixel correspondences automatically.

## Initial Model
A small custom CNN with residual blocks and a multi-scale feature pyramid that outputs:
- a local descriptor per sampled image location;
- a confidence score for each proposed match.

Train descriptors with a contrastive objective such as InfoNCE. Later, train confidence or uncertainty using downstream geometric quality.

## Main Deliverable
A reproducible system that compares:
- classical image features + geometric pose estimation;
- learned descriptors + geometric pose estimation;
- learned descriptors + learned confidence/uncertainty.

Evaluate matching quality, relative pose error, robustness to viewpoint/appearance change, and runtime.
