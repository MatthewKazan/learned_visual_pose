#include "pixel_to_pose.hpp"

#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "pinhole_camera.hpp"

namespace geometry::camera {

  // assert() compiles OUT under NDEBUG, which every optimized build defines --
  // so contract checks on data arriving from Python have to throw. pybind11
  // turns std::invalid_argument into a Python ValueError the caller can catch,
  // instead of a SIGABRT that kills the interpreter with no traceback.
  static void require(const bool ok, const char *what) {
    if (!ok) throw std::invalid_argument(what);
  }
  Eigen::Matrix3d eight_point_algorithm(const Eigen::MatrixX3d &ray_i, const Eigen::MatrixX3d &ray_j) {
    require(ray_i.rows() == ray_j.rows(),
            "eight_point_algorithm: ray_i and ray_j must have equal row counts");
    require(ray_i.rows() >= 8, "eight_point_algorithm: needs >= 8 correspondences");
    // u'u, u'v, u', v'u, v'v, v', u, v, 1]
    Eigen::MatrixXd A(ray_i.rows(), 9);
    for (Eigen::Index n = 0; n < ray_i.rows(); ++n) {
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> outer = ray_j.row(n).transpose() * ray_i.row(n);
      A.row(n) = Eigen::Map<Eigen::Matrix<double, 1, 9>>(outer.data());
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

    // get col of v associated with smallest singlular value (closest nullspace approx)
    Eigen::Matrix<double, 9, 1> es = svd.matrixV().col(8);

    Eigen::Matrix3d E = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(es.data());
    // Enforce rank 2 constraint
    Eigen::JacobiSVD<Eigen::Matrix3d> svdE(E, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Vector3d s = svdE.singularValues();
    s(2) = 0.0;
    auto ave = (s(0) + s(1)) / 2.0;
    s(0) = ave;
    s(1) = ave;
    E = svdE.matrixU() * s.asDiagonal() * svdE.matrixV().transpose();


    return E;
  }

  KabschFit kabsch_algorithm(const Eigen::MatrixX3d &point_i, const Eigen::MatrixX3d &point_j) {
    // Minimizes sum_k ||pj_k - (R pi_k + t)||^2. Returns T_ji: maps frame-i points into frame j.
    //
    // t separates exactly: the optimum maps centroid to centroid, t = mean_j - R mean_i,
    // so center both clouds and solve for R alone. Rotations preserve norms, so minimizing
    // the centered sum = maximizing sum_k pj'^T R pi', and via a^T M b = tr(M b a^T):
    //   maximize tr(R H),  H = sum_k pi'_k pj'_k^T  (outer product -- 3x3, not the scalar)
    // H = U S V^T gives tr(RH) = tr(M S) with M = V^T R U orthogonal, so |M_ii| <= 1 and
    // the max is M = I  ->  R = V U^T. Closed form, global optimum, no initial guess.
    //
    // det(V U^T) can be -1 (a mirror fits at least as well whenever sigma3 is weak --
    // routine on planar scenes). Fix flips only the smallest-sigma axis, diag(1,1,-1) with
    // JacobiSVD's descending sort. Negating the whole matrix also lands on det +1 but is a
    // ~180deg wrong rotation -- that bug shipped here once.
    //
    // Sigma diagnostics: sigma3 = 0 (coplanar) is fine -- handedness pins the third axis.
    // sigma2 ~ 0 (collinear) leaves rotation about the line undetermined.

    require(point_i.rows() == point_j.rows(),
            "kabsch_algorithm: point_i and point_j must have equal row counts");
    // fewer than 3 points cannot determine a rotation, and the resulting NaN
    // degeneracy silently passes any `< threshold` guard downstream
    require(point_i.rows() >= 3, "kabsch_algorithm: needs >= 3 points");

    const Eigen::Vector3d mean_i = point_i.colwise().mean();
    const Eigen::Vector3d mean_j = point_j.colwise().mean();

    const Eigen::MatrixX3d pi_c = point_i.rowwise() - mean_i.transpose();
    const Eigen::MatrixX3d pj_c = point_j.rowwise() - mean_j.transpose();

    const Eigen::Matrix3d H = pi_c.transpose() * pj_c;

    const Eigen::JacobiSVD<Eigen::Matrix3d> svdH(H, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Matrix3d U = svdH.matrixU();
    const Eigen::Matrix3d V = svdH.matrixV();
    Eigen::Matrix3d R = V * U.transpose();

    if (auto det = R.determinant(); det < 0.0) {
      Eigen::Matrix3d diag = Eigen::Matrix3d::Identity();
      diag(2,2) = -1.0;
      R = V * diag * U.transpose();
    }

    //                         3x1    3x3   3x1
    const Eigen::Vector3d t = mean_j - R * mean_i;

    return {PoseSE3(R, t), svdH.singularValues()(1) / svdH.singularValues()(0)};
  }

  // Function-local static: one generator for the whole library, fixed initial
  // state so runs replay. set_seed() is the only way to move it.
  static std::mt19937 &generator() {
    static std::mt19937 rng(0);
    return rng;
  }

  void set_seed(const unsigned seed) { generator().seed(seed); }

  static std::vector<Eigen::Index> random_points(const Eigen::Index num_points,
                                                 const int min_sample_size) {
    if (min_sample_size > num_points)
      throw std::invalid_argument("min_sample_size > num_points");
    std::uniform_int_distribution<size_t> dist(0, num_points - 1);
    std::unordered_set<Eigen::Index> idx;
    idx.reserve(min_sample_size);

    while (idx.size() < static_cast<size_t>(min_sample_size)) {
      idx.insert(dist(generator()));
    }

    return std::vector<Eigen::Index>(idx.begin(), idx.end());
  }

  std::vector<Eigen::Index> inlier_indices(const Eigen::MatrixX3d &point_i, const Eigen::MatrixX3d &point_j, const PoseSE3 &transform, const double inlier_threshold) {
    Eigen::MatrixX3d pj_est =  point_i * transform.rotation().matrix().transpose();
    pj_est = pj_est.rowwise() + transform.translation().transpose();

    const Eigen::MatrixX3d residuals = pj_est - point_j;

    std::vector<Eigen::Index> inlier_indicies;

    for (Eigen::Index i = 0; i < point_j.rows(); ++i) {
      if (residuals.row(i).norm() < inlier_threshold) {
        inlier_indicies.push_back(i);
      }
    }
    return inlier_indicies;
  }


  Consensus ransac_generic(
    const Eigen::Index num_points,
    const int min_sample_size,
    const ScoreFitFn &score_fit,
    const double desired_confidence,
    const int max_iterations
  ) {
    int num_iterations = max_iterations;
    int iterations_done = 0;

    std::vector<Eigen::Index> best_inliers;
    double best_w = 0.0;

    while (iterations_done < num_iterations) {
      auto candidate = score_fit(random_points(num_points, min_sample_size));
      // a degenerate sample still consumed an iteration
      iterations_done++;
      if (!candidate.has_value()) continue;

      auto &inliers = *candidate;
      if (!inliers.empty() && inliers.size() > best_inliers.size()) {
        best_w = static_cast<double>(inliers.size()) / static_cast<double>(num_points);
        // N = log(1-p) / log(1-w^s). Both logs are negative, so the quotient is
        // positive; min() because evidence may only ever SHRINK the budget.
        const auto candidate_iterations = static_cast<int>(
            std::log(1.0 - desired_confidence)
            / std::log(1.0 - std::pow(best_w, min_sample_size)));
        num_iterations = std::min(num_iterations, candidate_iterations);
        best_inliers = std::move(inliers);   // last use of `inliers`
      }
    }

    return {.inliers = best_inliers, .inlier_ratio = best_w};
  }

  RansacFit kabsch_ransac(
    const Eigen::MatrixX3d &point_i,
    const Eigen::MatrixX3d &point_j,
    const double inlier_threshold,
    const double degeneracy_threshold
  ) {
    require(point_i.rows() == point_j.rows(),
            "kabsch_ransac: point_i and point_j must have equal row counts");

    // idx -> pose. KabschFit unwraps here: degeneracy is consumed, not propagated.
    const auto fit = [&](const std::vector<Eigen::Index> &idx)
        -> std::optional<PoseSE3> {
      // fewer than 3 points makes the centroids and thus degeneracy NaN, and
      // NaN < threshold is FALSE -- the guard below would pass a garbage pose.
      if (idx.size() < 3) return std::nullopt;
      const KabschFit kab = kabsch_algorithm(point_i(idx, Eigen::placeholders::all),
                                             point_j(idx, Eigen::placeholders::all));
      if (kab.degeneracy < degeneracy_threshold) return std::nullopt;
      return kab.T_ji;
    };

    // pose -> who agrees with it
    const auto score = [&](const PoseSE3 &T) {
      return inlier_indices(point_i, point_j, T, inlier_threshold);
    };

    const auto score_fit = [&](const std::vector<Eigen::Index> &idx)
        -> std::optional<std::vector<Eigen::Index>> {
      if (const auto T = fit(idx)) return score(*T);
      return std::nullopt;
    };

    const Consensus consensus = ransac_generic(point_i.rows(), 3, score_fit);

    // the minimal-sample fits were only search probes -- 3 points, no averaging.
    // This refit over the whole consensus set is where the accuracy comes from.
    const auto T = fit(consensus.inliers);
    if (!T) return {.T_ji = PoseSE3(), .inliers = {}, .inlier_ratio = 0.0};

    const auto inliers = score(*T);
    return {.T_ji = *T,
            .inliers = inliers,
            .inlier_ratio = static_cast<double>(inliers.size())
                            / static_cast<double>(point_i.rows())};
  }

  std::vector<Eigen::Index> farthest_point_sample(const Eigen::MatrixX3d &points,
                                                  const Eigen::Index num_points_to_sample) {
    require(points.cols() == 3, "farthest_point_sample: points must be Nx3");
    require(num_points_to_sample >= 1,
            "farthest_point_sample: num_points_to_sample must be >= 1");
    const Eigen::Index num_points = points.rows();

    // "give me all of them" is a legitimate request (the top of a K sweep), so
    // clamp rather than abort -- an assert here would kill the Python process.
    if (num_points_to_sample >= num_points) {
      std::vector<Eigen::Index> all(num_points);
      std::iota(all.begin(), all.end(), 0);
      return all;
    }

    std::vector<double> dist_to_nearest_sample(num_points);

    auto first_set = random_points(num_points, 1);
    Eigen::Index first_point = *first_set.begin();

    for (Eigen::Index i = 0; i < num_points; ++i) {
      dist_to_nearest_sample[i] = (points.row(i) - points.row(first_point)).norm();
    }

    std::vector<Eigen::Index> sampled;
    sampled.reserve(num_points_to_sample);
    sampled.push_back(first_point);

    while (static_cast<Eigen::Index>(sampled.size()) < num_points_to_sample) {
      const Eigen::Index next_p_idx = std::distance(
        dist_to_nearest_sample.begin(),
        std::ranges::max_element(dist_to_nearest_sample)
      );

      sampled.push_back(next_p_idx);
      auto next_p = points.row(next_p_idx);

      // a chosen point's distance drops to 0, so max_element cannot re-pick it
      for (Eigen::Index i = 0; i < num_points; ++i) {
        dist_to_nearest_sample[i] = std::min(dist_to_nearest_sample[i],
                                             (points.row(i) - next_p).norm());
      }
    }
    return sampled;
  }

}
