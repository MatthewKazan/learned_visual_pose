#pragma once
#include <Eigen/Dense>
#include <functional>
#include <optional>
#include <vector>

namespace geometry::camera {

  /**
   * Take in list of corresponding rays and return the essential matrix E.
   *
   * E satisfies: ray_j.T * E * ray_i = 0
   *
   * @param ray_i [[u, v, 1], ...] normalized pixel coords for all correspondences in image i
   * @param ray_j [[u', v', 1], ...] normalized pixel coords for all correspondences in image j
   * @return
   */
  Eigen::Matrix3d eight_point_algorithm(const Eigen::MatrixX3d &ray_i, const Eigen::MatrixX3d &ray_j);

  struct KabschFit {
    Eigen::Matrix4d T_ji;
    /// sigma2/sigma1. Near zero means collinear input: rotation about that line
    /// is undetermined and T_ji is arbitrary. Coplanar input is NOT flagged --
    /// it is a valid case here, unlike for the essential matrix.
    double degeneracy;
  };

  /// What RANSAC itself produces: the largest agreeing subset, and nothing about
  /// the model. Turning a consensus set into a pose is the caller's job, which is
  /// why this solver works for any model -- pose, homography, plane, line.
  struct Consensus {
    std::vector<Eigen::Index> inliers;
    double inlier_ratio;
  };

  struct RansacFit {
    Eigen::Matrix4d T_ji;
    /// Indices into the input clouds, from the post-refit classification.
    /// Empty (and inlier_ratio 0) means no usable consensus was found.
    std::vector<Eigen::Index> inliers;
    /// Feeds the information matrix on every pose graph edge.
    double inlier_ratio;
  };

  /**
   * Seed every randomised algorithm in this library (currently RANSAC sampling).
   *
   * The generator starts at a FIXED state, so results are already repeatable
   * without calling this -- a measurement you cannot repeat is not a
   * measurement. Call it to study sampling variance deliberately.
   *
   * NOT thread safe: one generator is shared library-wide, so concurrent solves
   * race on it. Fine while Python drives one solve at a time.
   */
  void set_seed(unsigned seed);

  /// idx -> the consensus set for whatever model those samples imply, or nullopt
  /// if the sample was degenerate for that model. Fit and score are collapsed
  /// because the loop only ever compares inlier COUNTS.
  using ScoreFitFn = std::function<std::optional<std::vector<Eigen::Index>>(
      const std::vector<Eigen::Index> &)>;

  /**
   * Run Kabsch Algorithm to find the best transformation to align two pointclouds
   *
   * @param point_i point cloud i (Nx3)
   * @param point_j point cloud j (Nx3)
   *
   * @return 4x4 Transformation matrix aligning the point clouds and σ2/σ1 ratio
   */
  KabschFit kabsch_algorithm(const Eigen::MatrixX3d &point_i, const Eigen::MatrixX3d &point_j);

  std::vector<Eigen::Index> inlier_indices(const Eigen::MatrixX3d &point_i,
                                           const Eigen::MatrixX3d &point_j,
                                           const Eigen::Matrix4d &transform,
                                           double inlier_threshold);

  /**
   * Model-agnostic RANSAC: sample, score, keep the largest consensus.
   *
   * Knows nothing about geometry -- `score_fit` closes over the data, so the
   * same loop serves rigid fits, homographies, or anything else. min_sample_size
   * is the model's minimal set; it drives N = log(1-p)/log(1-w^s).
   */
  Consensus ransac_generic(Eigen::Index num_points,
                           int min_sample_size,
                           const ScoreFitFn &score_fit,
                           double desired_confidence = 0.99,
                           int max_iterations = 50);

  /**
   * Kabsch inside ransac_generic, plus the consensus refit. 3-point minimal samples.
   *
   * @param inlier_threshold      METRES -- set from depth noise at working range,
   *                              not tuned as a dimensionless number.
   * @param degeneracy_threshold  floor on sigma2/sigma1; below it the sample is
   *                              collinear and its rotation is undetermined.
   */
  RansacFit kabsch_ransac(const Eigen::MatrixX3d &point_i,
                          const Eigen::MatrixX3d &point_j,
                          double inlier_threshold = 0.05,
                          double degeneracy_threshold = 1e-3);


}
