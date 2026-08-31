#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <gtest/gtest.h>
#include <geometry.hpp>
#include <algorithm>
#include <cmath>

#include "pixel_to_pose.hpp"

// Synthetic epipolar test data. Each row is one correspondence [x, y, 1] in
// normalized (calibrated) coords, so K never enters. Rows of uv_i and uv_j at
// the same index are the same 3D point seen from camera i and camera j.
//
// Generated from known (R, t) by projecting 9 non-coplanar points, so
// uv_j.row(n) * E * uv_i.row(n).transpose() == 0 exactly, and E = hat(t) * R.
//
// Comparing a recovered E to these: the SVD returns a UNIT-NORM vector and the
// sign is arbitrary, so normalize both by Frobenius norm and allow either sign.
// Asserting on the singular values instead (sigma3 == 0, sigma1 == sigma2)
// sidesteps both ambiguities.

using Correspondences = Eigen::Matrix<double, 9, 3>;

// ---- A: R = I, t = (-1, 0, 0). Pure sideways motion. ----
// Every pair shares its y, i.e. horizontal epipolar lines -- checkable by eye.
const Correspondences uv_i_sideways{
    { 0.00,  0.00, 1},
    { 1.00,  0.00, 1},
    { 0.00,  0.50, 1},
    { 1.00, -0.50, 1},
    {-0.25,  0.25, 1},
    { 0.50,  0.50, 1},
    {-0.40, -0.20, 1},
    { 0.20,  0.60, 1},
    { 0.75, -0.50, 1},
};
const Correspondences uv_j_sideways{
    {-1.00,  0.00, 1},
    { 0.00,  0.00, 1},
    {-0.50,  0.50, 1},
    { 0.50, -0.50, 1},
    {-0.50,  0.25, 1},
    { 0.25,  0.50, 1},
    {-0.60, -0.20, 1},
    { 0.00,  0.60, 1},
    { 0.50, -0.50, 1},
};
const Eigen::Matrix3d E_sideways{
    {0,  0, 0},
    {0,  0, 1},
    {0, -1, 0},
};

// ---- B: R = Rz(90 deg), t = (-1, 0, 0). ----
// Exact 0/+-1 rotation entries, so a transposed R or swapped axis fails loudly
// rather than subtly.
const Correspondences uv_i_rotated{
    { 0.00,  0.00, 1},
    { 1.00,  0.00, 1},
    { 0.00,  0.50, 1},
    { 1.00, -0.50, 1},
    {-0.25,  0.25, 1},
    { 0.50,  0.50, 1},
    {-0.40, -0.20, 1},
    { 0.20,  0.60, 1},
    { 0.75, -0.50, 1},
};
const Correspondences uv_j_rotated{
    {-1.00,  0.00, 1},
    {-1.00,  1.00, 1},
    {-1.00,  0.00, 1},
    { 0.00,  1.00, 1},
    {-0.50, -0.25, 1},
    {-0.75,  0.50, 1},
    { 0.00, -0.40, 1},
    {-0.80,  0.20, 1},
    { 0.25,  0.75, 1},
};
const Eigen::Matrix3d E_rotated{
    { 0, 0, 0},   // row is zero because t_z == 0
    { 0, 0, 1},
    {-1, 0, 0},
};

// ---- D: R = Rz(90 deg), t = (-2, 1, 1). PRIMARY TEST. ----
// A and B both have t_z == 0, which zeroes the whole first row of hat(t) -- a
// bug corrupting row 0 of E passes both silently. Here every entry is live.
// Epipole in image i is the null vector of E: (1, 2, 1), i.e. ray coords (1, 2).
const Correspondences uv_i_general{
    { 0.00,  0.00, 1},
    { 1.00,  0.00, 1},
    { 0.00,  0.50, 1},
    { 1.00, -0.50, 1},
    {-0.25,  0.25, 1},
    { 0.50,  0.50, 1},
    {-0.40, -0.20, 1},
    { 0.20,  0.60, 1},
    { 0.75, -0.50, 1},
};
const Correspondences uv_j_general{
    {-1.0,        0.5,       1},
    {-1.0,        1.0,       1},
    {-1.0,        1.0 / 3.0, 1},
    {-1.0 / 3.0,  1.0,       1},
    {-0.6,        0.0,       1},
    {-0.8,        0.6,       1},
    {-1.0 / 6.0, -1.0 / 6.0, 1},
    {-5.0 / 6.0,  1.0 / 3.0, 1},
    { 0.0,        0.8,       1},
};
const Eigen::Matrix3d E_general{
    {-1,  0, 1},
    { 0, -1, 2},
    {-2,  1, 0},
};
// singular values are (2.449490, 2.449490, 0) -- sigma1 / sigma2 == 1 exactly
const Eigen::Vector3d epipole_i_general{1.0, 2.0, 1.0};

// ---- C: DEGENERATE. Same (R, t) as B, all 9 points on the plane Z = 3. ----
// The nullspace of A is 3-dimensional, not 1, so there is no unique E and the
// SVD returns an arbitrary vector from that space without complaint. Coplanar
// scenes are the most common real failure -- any frame facing a wall or floor.
const Correspondences uv_i_coplanar{
    { 0.0,       0.0,       1},
    { 1.0 / 3.0, 0.0,       1},
    { 0.0,       1.0 / 3.0, 1},
    { 2.0 / 3.0,-1.0 / 3.0, 1},
    {-1.0 / 3.0, 1.0 / 3.0, 1},
    { 2.0 / 3.0, 2.0 / 3.0, 1},
    {-2.0 / 3.0,-1.0 / 3.0, 1},
    { 1.0 / 3.0, 1.0,       1},
    { 1.0,      -2.0 / 3.0, 1},
};
const Correspondences uv_j_coplanar{
    {-1.0 / 3.0,  0.0,       1},
    {-1.0 / 3.0,  1.0 / 3.0, 1},
    {-2.0 / 3.0,  0.0,       1},
    { 0.0,        2.0 / 3.0, 1},
    {-2.0 / 3.0, -1.0 / 3.0, 1},
    {-1.0,        2.0 / 3.0, 1},
    { 0.0,       -2.0 / 3.0, 1},
    {-4.0 / 3.0,  1.0 / 3.0, 1},
    { 1.0 / 3.0,  1.0,       1},
};

// ---- E: same (R, t) as D, plus ~1.5px of Gaussian noise (sigma = 1.5/320). ----
// Exists to make the rank-2 projection observable. The noise-free sets above
// come out satisfying (sigma, sigma, 0) already, so they cannot tell whether
// the projection step is present. With noise the raw nullspace solution has
// sigma = (0.712186, 0.701947, 0.007880) -- neither rank 2 nor equal-valued.
// Projecting to (1, 1, 0) also moves E closer to truth, 0.01693 -> 0.01313,
// so the step earns its place rather than just tidying the singular values.
const Correspondences uv_i_noisy{
    { 0.000006,  0.001400, 1},
    { 0.998715, -0.004175, 1},
    {-0.002131,  0.495352, 1},
    { 1.000282, -0.493718, 1},
    {-0.252307,  0.247092, 1},
    { 0.502296,  0.501673, 1},
    {-0.399506, -0.204362, 1},
    { 0.199863,  0.603259, 1},
    { 0.743699, -0.502145, 1},
};
const Correspondences uv_j_noisy{
    {-1.008912,  0.493955, 1},
    {-1.008633,  0.998898, 1},
    {-1.005941,  0.334605, 1},
    {-0.332598,  0.999124, 1},
    {-0.611797, -0.002525, 1},
    {-0.800227,  0.600531, 1},
    {-0.173840, -0.168906, 1},
    {-0.837920,  0.329542, 1},
    { 0.004973,  0.796215, 1},
};

namespace {
  using geometry::camera::eight_point_algorithm;

  // E is only defined up to scale AND sign -- the nullspace vector comes back
  // unit-norm with an arbitrary sign. Normalize, then take the closer of the
  // two sign choices.
  double e_distance(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
    const Eigen::Matrix3d an = a / a.norm();
    const Eigen::Matrix3d bn = b / b.norm();
    return std::min((an - bn).norm(), (an + bn).norm());
  }

  double max_residual(const Eigen::Matrix3d &E,
                      const Eigen::MatrixX3d &ray_i, const Eigen::MatrixX3d &ray_j) {
    double worst = 0.0;
    for (Eigen::Index n = 0; n < ray_i.rows(); ++n) {
      worst = std::max(worst, std::abs((ray_j.row(n) * E * ray_i.row(n).transpose()).value()));
    }
    return worst;
  }
}

// Every row, not just row 0. Row 0 of every dataset is the point on the optical
// axis, so x_i is (0, 0, 1) and terms vanish there that do not vanish elsewhere
// -- a solver that got rows 1-8 wrong would still pass a row-0-only check.
TEST(TestPixelToPose, SatisfiesEpipolarConstraint) {
  EXPECT_NEAR(max_residual(eight_point_algorithm(uv_i_sideways, uv_j_sideways),
                           uv_i_sideways, uv_j_sideways), 0.0, 1e-9);
  EXPECT_NEAR(max_residual(eight_point_algorithm(uv_i_rotated, uv_j_rotated),
                           uv_i_rotated, uv_j_rotated), 0.0, 1e-9);
  EXPECT_NEAR(max_residual(eight_point_algorithm(uv_i_general, uv_j_general),
                           uv_i_general, uv_j_general), 0.0, 1e-9);
}

// Residuals alone are not enough: a degenerate input produces an E that
// satisfies every equation and is still the wrong matrix. Compare against the
// (R, t) the data was generated from.
TEST(TestPixelToPose, RecoversKnownE) {
  EXPECT_NEAR(e_distance(eight_point_algorithm(uv_i_sideways, uv_j_sideways),
                         E_sideways), 0.0, 1e-9);
  EXPECT_NEAR(e_distance(eight_point_algorithm(uv_i_rotated, uv_j_rotated),
                         E_rotated), 0.0, 1e-9);
  EXPECT_NEAR(e_distance(eight_point_algorithm(uv_i_general, uv_j_general),
                         E_general), 0.0, 1e-9);
}

// Eight is the minimum: 9 unknowns, homogeneous so 8 of them are free.
TEST(TestPixelToPose, EightPointsAreEnough) {
  const Eigen::Matrix3d E = eight_point_algorithm(uv_i_general.topRows(8),
                                                  uv_j_general.topRows(8));
  EXPECT_NEAR(e_distance(E, E_general), 0.0, 1e-9);
}

// Each correspondence is an independent row of A, so permuting them permutes
// rows of A and leaves its nullspace untouched.
TEST(TestPixelToPose, IndependentOfCorrespondenceOrder) {
  const Correspondences ri = uv_i_general.colwise().reverse();
  const Correspondences rj = uv_j_general.colwise().reverse();
  EXPECT_NEAR(e_distance(eight_point_algorithm(ri, rj), E_general), 0.0, 1e-9);
}

// A ray is a direction: s*x names the same ray for any s > 0, and the third
// component being 1 is only a convention. Scaling a row scales that equation,
// and a scaled zero is still zero.
TEST(TestPixelToPose, IndependentOfRayScale) {
  const double s[9] = {1.0, 2.0, 0.5, 3.0, 0.25, 4.0, 1.5, 0.75, 5.0};
  Correspondences si = uv_i_general, sj = uv_j_general;
  for (int n = 0; n < 9; ++n) {
    si.row(n) *= s[n];
    sj.row(n) *= s[8 - n];
  }
  EXPECT_NEAR(e_distance(eight_point_algorithm(si, sj), E_general), 0.0, 1e-9);
}

// x_j' E x_i == 0 is the same scalar as x_i' E' x_j == 0, so swapping which
// image is which transposes E. Catches an i/j mix-up inside the solver, which
// is otherwise invisible -- both orders produce a plausible-looking matrix.
TEST(TestPixelToPose, SwappingImagesTransposesE) {
  const Eigen::Matrix3d E = eight_point_algorithm(uv_i_general, uv_j_general);
  const Eigen::Matrix3d E_swapped = eight_point_algorithm(uv_j_general, uv_i_general);
  EXPECT_NEAR(e_distance(E_swapped, E.transpose()), 0.0, 1e-9);
}

// E * e == 0 where e is the direction from camera i toward camera j, because
// E = hat(t) * R and hat(t) * t == 0. This is where VIS-012 gets the epipole.
TEST(TestPixelToPose, EpipoleIsTheNullVector) {
  const Eigen::Matrix3d E = eight_point_algorithm(uv_i_general, uv_j_general);
  EXPECT_NEAR((E * epipole_i_general).norm(), 0.0, 1e-9);
}

// FAILS until eight_point_algorithm projects its result onto the essential
// manifold: SVD, replace the singular values with (s, s, 0), multiply back.
// Both assertions hold for every essential matrix and neither is affected by
// the scale or sign ambiguity.
TEST(TestPixelToPose, ProjectsOntoEssentialManifold) {
  const Eigen::Matrix3d E = eight_point_algorithm(uv_i_noisy, uv_j_noisy);
  const Eigen::Vector3d sv = E.jacobiSvd().singularValues();

  EXPECT_NEAR(sv(2), 0.0, 1e-9) << "E must be rank 2";
  EXPECT_NEAR(sv(0), sv(1), 1e-9) << "E's two nonzero singular values must be equal";
}

// ~1.5px of noise costs about 1.3% -- graceful, not catastrophic. Guards
// against a change that still passes the noise-free tests but degrades badly
// on real data, which is the only kind this solver will ever see.
TEST(TestPixelToPose, NoisyInputStaysClose) {
  const Eigen::Matrix3d E = eight_point_algorithm(uv_i_noisy, uv_j_noisy);
  EXPECT_LT(e_distance(E, E_general), 0.05);
}

// Coplanar points leave A with a 3-dimensional nullspace, so the SVD returns an
// arbitrary member of it. Note what still holds: the result is a structurally
// perfect essential matrix. Rank and singular-value checks CANNOT detect this;
// only the rank of A can. The recovered E itself is not asserted on because it
// differs between linear algebra libraries -- Eigen and numpy pick different
// vectors out of the same nullspace.
TEST(TestPixelToPose, DegenerateInputStillLooksValid) {
  const Eigen::Matrix3d E = eight_point_algorithm(uv_i_coplanar, uv_j_coplanar);
  const Eigen::Vector3d sv = E.jacobiSvd().singularValues();

  EXPECT_NEAR(sv(2), 0.0, 1e-9);
  EXPECT_NEAR(sv(0), sv(1), 1e-9);
}
// ─────────────────────────── Kabsch ───────────────────────────
//
// Assumed contract, since the header does not state it: kabsch_algorithm(P_i,
// P_j).T_ji returns T_ji, the transform carrying frame i's points into frame j, so
// P_j == R * P_i + t. Same direction as eight_point_algorithm(ray_i, ray_j).
// If you settle on the other direction, every expectation below flips.
//
// Nothing here is defined up to scale or sign, unlike E. The output is a rigid
// transform: unique, full rank, no ambiguity. So these assert to 1e-12 and
// compare R and t directly rather than through a distance function.
//
// All data below was generated from R = Rz(90 deg), t = (-2, 1, 1) and verified
// numerically before being written down.

using PointCloud = Eigen::Matrix<double, 9, 3>;

const Eigen::Matrix3d R_kabsch{
    {0, -1, 0},
    {1,  0, 0},
    {0,  0, 1},
};
const Eigen::Vector3d t_kabsch{-2, 1, 1};

// ---- K-A: general. Centred cloud has rank 3, singular values (5.07, 4.36,
// 3.39), so the fit is fully determined in every direction. ----
const PointCloud kabsch_i_general{
    { 0,  0, 3},
    { 1,  0, 3},
    { 0,  1, 3},
    { 2, -1, 4},
    {-1,  1, 5},
    { 2,  2, 4},
    {-2, -1, 6},
    { 1,  3, 5},
    { 3, -2, 7},
};
const PointCloud kabsch_j_general{
    {-2,  1, 4},
    {-2,  2, 4},
    {-3,  1, 4},
    {-1,  3, 5},
    {-3,  0, 6},
    {-4,  3, 5},
    {-1, -1, 7},
    {-5,  2, 6},
    { 0,  4, 8},
};

// ---- K-B: COPLANAR, all z = 3. This is the case that destroys the eight-point
// algorithm -- see uv_i_coplanar above -- and it does not touch Kabsch. Three
// non-collinear points already pin all six DOF; a flat wall is fully
// constraining once you know how far away it is. Cross-covariance singular
// values are (22, 18, 0): the zero costs nothing, because the reflection
// correction resolves the only ambiguity it creates. ----
const PointCloud kabsch_i_coplanar{
    { 0,  0, 3},
    { 1,  0, 3},
    { 0,  1, 3},
    { 2, -1, 3},
    {-1,  1, 3},
    { 2,  2, 3},
    {-2, -1, 3},
    { 1,  3, 3},
    { 3, -2, 3},
};
const PointCloud kabsch_j_coplanar{
    {-2,  1, 4},
    {-2,  2, 4},
    {-3,  1, 4},
    {-1,  3, 4},
    {-3,  0, 4},
    {-4,  3, 4},
    {-1, -1, 4},
    {-5,  2, 4},
    { 0,  4, 4},
};

// ---- K-C: DEGENERATE. All 9 points on the line (k, k, k+3). Rotation about
// that line is unconstrained; the cross-covariance is rank 1, singular values
// (180, 0, 0). The trap is identical in shape to the coplanar one above:
// the returned R is orthonormal with det +1 and maps every point exactly, and
// is still not the rotation the data came from. Only the rank of the
// cross-covariance can detect it. ----
const PointCloud kabsch_i_collinear{
    {-4, -4, -1},
    {-3, -3,  0},
    {-2, -2,  1},
    {-1, -1,  2},
    { 0,  0,  3},
    { 1,  1,  4},
    { 2,  2,  5},
    { 3,  3,  6},
    { 4,  4,  7},
};
const PointCloud kabsch_j_collinear{
    { 2, -3, 0},
    { 1, -2, 1},
    { 0, -1, 2},
    {-1,  0, 3},
    {-2,  1, 4},
    {-3,  2, 5},
    {-4,  3, 6},
    {-5,  4, 7},
    {-6,  5, 8},
};

// ---- K-D: REFLECTION TRAP. Points on z = 0, mirrored in x. Two orthogonal
// matrices map i onto j with ZERO residual: diag(-1, 1, 1), a reflection, and
// Ry(180) = diag(-1, 1, -1), a rotation. Both fit perfectly -- verified, both
// residuals are exactly 0 -- so no residual or orthonormality check can tell
// them apart. Only the determinant can. This is the fixture that catches a
// missing diag(1, 1, sign(det)) correction, and nothing else in this file
// does. ----
const PointCloud kabsch_i_reflect{
    { 0,  0, 0},
    { 1,  0, 0},
    { 0,  1, 0},
    { 2, -1, 0},
    {-1,  1, 0},
    { 2,  2, 0},
    {-2, -1, 0},
    { 1,  3, 0},
    { 3, -2, 0},
};
const PointCloud kabsch_j_reflect{
    { 0,  0, 0},
    {-1,  0, 0},
    { 0,  1, 0},
    {-2, -1, 0},
    { 1,  1, 0},
    {-2,  2, 0},
    { 2, -1, 0},
    {-1,  3, 0},
    {-3, -2, 0},
};
const Eigen::Matrix3d R_reflect_expected{   // Ry(180), the det +1 answer
    {-1, 0,  0},
    { 0, 1,  0},
    { 0, 0, -1},
};

// ---- K-E: K-A plus Gaussian noise, sigma = 0.01 m. Costs 0.176 deg and
// 1.09 cm, i.e. graceful. Also the only fixture where orthonormality is a live
// question: noise makes the cross-covariance a general matrix, and U * V^T is
// still exactly orthogonal only if the SVD is used correctly. ----
const PointCloud kabsch_i_noisy{
    { 0.000012,  0.002987, 2.997259},
    { 0.991094, -0.004547, 2.990084},
    { 0.000601,  1.013402, 2.995078},
    { 1.993795, -0.995102, 4.003569},
    {-0.998946,  0.990695, 4.999707},
    { 2.006953,  1.986558, 3.995424},
    {-2.019012, -1.012895, 5.981583},
    { 0.997649,  2.987326, 5.002713},
    { 3.001568, -2.001869, 6.974832},
};
const PointCloud kabsch_j_noisy{
    {-2.005387,  0.999515, 4.001133},
    {-2.015301,  1.995222, 3.990215},
    {-3.008088,  1.010609, 3.991925},
    {-1.000325,  3.008844, 4.994164},
    {-3.001117,  0.001105, 6.000638},
    {-4.012251,  3.000761, 5.013588},
    {-1.015471, -0.991406, 7.001194},
    {-5.006415,  2.020004, 6.007623},
    {-0.011993,  4.000745, 8.005767},
};

namespace {
  using geometry::camera::kabsch_algorithm;

  Eigen::Matrix3d rotation_of(const Eigen::Matrix4d &T) { return T.topLeftCorner<3, 3>(); }
  Eigen::Vector3d translation_of(const Eigen::Matrix4d &T) { return T.topRightCorner<3, 1>(); }

  // Worst per-point error of P_j vs T * P_i, in metres.
  double worst_point_error(const Eigen::Matrix4d &T,
                           const Eigen::MatrixX3d &P_i, const Eigen::MatrixX3d &P_j) {
    const Eigen::MatrixX3d mapped =
        (P_i * rotation_of(T).transpose()).rowwise() + translation_of(T).transpose();
    return (mapped - P_j).rowwise().norm().maxCoeff();
  }

  double angle_between(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
    const double c = ((a * b.transpose()).trace() - 1.0) / 2.0;
    return std::acos(std::clamp(c, -1.0, 1.0));
  }
}

// The whole point of the 3D-3D formulation: noise-free input is recovered to
// machine precision, not "close". The eight-point algorithm has no equivalent
// assertion because E is only defined up to scale and sign.
TEST(TestKabsch, RecoversKnownTransformExactly) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_general, kabsch_j_general).T_ji;

  EXPECT_LT((rotation_of(T) - R_kabsch).norm(), 1e-12);
  EXPECT_LT((translation_of(T) - t_kabsch).norm(), 1e-12);
}

// Every point, not just the fit's own cost. Row 0 is the origin of the cloud,
// where a translation-only bug is invisible.
TEST(TestKabsch, MapsEveryPointExactly) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_general, kabsch_j_general).T_ji;
  EXPECT_NEAR(worst_point_error(T, kabsch_i_general, kabsch_j_general), 0.0, 1e-12);
}

// A Matrix4d has room to be malformed. The bottom row is not decoration: chain
// two transforms with a wrong one and the error is silent and cumulative.
TEST(TestKabsch, ReturnsAWellFormedRigidTransform) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_noisy, kabsch_j_noisy).T_ji;
  const Eigen::Matrix3d R = rotation_of(T);

  EXPECT_LT((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
  EXPECT_LT((T.bottomRows<1>() - Eigen::RowVector4d(0, 0, 0, 1)).norm(), 1e-12);
}

// THE test. Both diag(-1, 1, 1) and Ry(180) map this data with zero residual,
// so residual and orthonormality both pass on the reflection. A missing
// diag(1, 1, sign(det)) correction fails here and nowhere else in this file.
TEST(TestKabsch, RejectsReflections) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_reflect, kabsch_j_reflect).T_ji;

  EXPECT_NEAR(rotation_of(T).determinant(), 1.0, 1e-12) << "returned a reflection";
  EXPECT_LT((rotation_of(T) - R_reflect_expected).norm(), 1e-12);
  EXPECT_NEAR(worst_point_error(T, kabsch_i_reflect, kabsch_j_reflect), 0.0, 1e-12);
}

// Three non-collinear points determine a rigid transform completely. Their
// cross-covariance is rank 2, singular values (1, 0.333, 0) -- the zero is the
// same reflection ambiguity as K-D, resolved the same way.
TEST(TestKabsch, ThreePointsAreEnough) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_general.topRows(3),
                                             kabsch_j_general.topRows(3)).T_ji;

  EXPECT_LT((rotation_of(T) - R_kabsch).norm(), 1e-12);
  EXPECT_LT((translation_of(T) - t_kabsch).norm(), 1e-12);
}

// The cross-covariance is a SUM over correspondences, and addition commutes.
TEST(TestKabsch, IndependentOfCorrespondenceOrder) {
  const PointCloud ri = kabsch_i_general.colwise().reverse();
  const PointCloud rj = kabsch_j_general.colwise().reverse();
  const Eigen::Matrix4d T = kabsch_algorithm(ri, rj).T_ji;

  EXPECT_LT((rotation_of(T) - R_kabsch).norm(), 1e-12);
  EXPECT_LT((translation_of(T) - t_kabsch).norm(), 1e-12);
}

// T_ij == T_ji^-1, exactly -- so R flips to R^T and t to -R^T t. Stronger than
// the eight-point's swap test, which only recovers E up to a transpose because
// E carries no translation magnitude. Catches an i/j mix-up, which is
// otherwise invisible: both orders produce a plausible rigid transform.
TEST(TestKabsch, SwappingCloudsInvertsTheTransform) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_general, kabsch_j_general).T_ji;
  const Eigen::Matrix4d T_swapped = kabsch_algorithm(kabsch_j_general, kabsch_i_general).T_ji;

  EXPECT_LT((T_swapped - T.inverse()).norm(), 1e-12);
}

// Kabsch centres both clouds before forming the cross-covariance, so a common
// shift cannot reach the rotation. Fails loudly if the centring is dropped or
// applied to only one cloud, which is the other classic bug in this algorithm.
TEST(TestKabsch, CommonShiftLeavesRotationUnchanged) {
  const Eigen::Vector3d shift{10, -5, 7};
  const PointCloud si = kabsch_i_general.rowwise() + shift.transpose();
  const PointCloud sj = kabsch_j_general.rowwise() + shift.transpose();
  const Eigen::Matrix4d T = kabsch_algorithm(si, sj).T_ji;

  EXPECT_LT((rotation_of(T) - R_kabsch).norm(), 1e-12);
  // the same rigid motion, re-expressed about a shifted origin
  EXPECT_LT((translation_of(T) - (t_kabsch + shift - R_kabsch * shift)).norm(), 1e-12);
}

// Coplanar input destroys the eight-point algorithm (DegenerateInputStillLooksValid
// above) and is exact here. This is the whole argument for the 3D-3D
// formulation, stated as a test: depth removes the degeneracy rather than
// improving the conditioning.
TEST(TestKabsch, CoplanarInputIsExact) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_coplanar, kabsch_j_coplanar).T_ji;

  EXPECT_LT((rotation_of(T) - R_kabsch).norm(), 1e-12);
  EXPECT_LT((translation_of(T) - t_kabsch).norm(), 1e-12);
}

// Kabsch's own degeneracy, and it is COLLINEAR, not coplanar. Rotation about
// the line is free, so the cross-covariance is rank 1 and the SVD returns an
// arbitrary member of the remaining family. Note what still holds: orthonormal,
// det +1, and every point mapped exactly. Verified numerically -- the returned
// R differs from truth by a full radian while the worst point error is 4e-15.
// No structural check can detect this; only the rank of the cross-covariance
// can. The recovered R is not asserted on, because Eigen and numpy pick
// different members of the same degenerate subspace.
TEST(TestKabsch, CollinearInputStillLooksValid) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_collinear, kabsch_j_collinear).T_ji;
  const Eigen::Matrix3d R = rotation_of(T);

  EXPECT_LT((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
  EXPECT_NEAR(worst_point_error(T, kabsch_i_collinear, kabsch_j_collinear), 0.0, 1e-9);
}

// sigma = 1 cm of noise costs 0.176 deg and 1.09 cm -- graceful, not
// catastrophic. Guards against a change that passes every exact test above and
// degrades badly on real data, which is the only kind this will ever see.
TEST(TestKabsch, NoisyInputStaysClose) {
  const Eigen::Matrix4d T = kabsch_algorithm(kabsch_i_noisy, kabsch_j_noisy).T_ji;

  EXPECT_LT(angle_between(rotation_of(T), R_kabsch), 0.5 * M_PI / 180.0);
  EXPECT_LT((translation_of(T) - t_kabsch).norm(), 0.03);
}
// ---------------------------------------------------------------------------
// RANSAC
//
// ransac_generic is deliberately blind to geometry, so most of it is testable
// with a fake score_fit: no clouds, no Eigen, deterministic. Only the last two
// tests need real points.
// ---------------------------------------------------------------------------

namespace {
  using geometry::camera::Consensus;
  using geometry::camera::kabsch_ransac;
  using geometry::camera::ransac_generic;

  // Pretends indices [0, n) are the planted consensus set: a sample drawn
  // entirely from inside it "fits" and reports the whole set, anything else
  // reports a small bogus set. Mirrors what a real fitter does without geometry.
  geometry::camera::ScoreFitFn planted(Eigen::Index n, int *calls = nullptr) {
    return [n, calls](const std::vector<Eigen::Index> &idx)
        -> std::optional<std::vector<Eigen::Index>> {
      if (calls) ++*calls;
      std::vector<Eigen::Index> out;
      const bool clean = std::all_of(idx.begin(), idx.end(),
                                     [n](Eigen::Index i) { return i < n; });
      for (Eigen::Index i = 0; i < (clean ? n : 2); ++i) out.push_back(i);
      return out;
    };
  }

  Eigen::Matrix4d transform_of(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = t;
    return T;
  }
}

TEST(TestRansacGeneric, FindsPlantedConsensus) {
  const Consensus c = ransac_generic(200, 3, planted(150));

  EXPECT_EQ(c.inliers.size(), 150u);
  EXPECT_NEAR(c.inlier_ratio, 0.75, 1e-12);
}

// A sample can only be clean if all 3 indices land inside the consensus set, so
// with w = 0.9 the loop should succeed early and cut its own budget well below
// max_iterations. This is the N = log(1-p)/log(1-w^s) adaptation, observed
// rather than recomputed.
TEST(TestRansacGeneric, AdaptiveBudgetShrinks) {
  int calls = 0;
  ransac_generic(100, 3, planted(90, &calls), 0.99, 500);

  EXPECT_LT(calls, 100);
  EXPECT_GT(calls, 0);
}

// A fitter that rejects everything must not produce a consensus, and must not
// spin: every degenerate sample still consumes an iteration.
TEST(TestRansacGeneric, AllDegenerateTerminatesEmpty) {
  const auto never = [](const std::vector<Eigen::Index> &)
      -> std::optional<std::vector<Eigen::Index>> { return std::nullopt; };
  const Consensus c = ransac_generic(50, 3, never);

  EXPECT_TRUE(c.inliers.empty());
  EXPECT_EQ(c.inlier_ratio, 0.0);
}

// The point of the whole ticket: least squares lets one gross outlier dominate,
// RANSAC does not. Same data through kabsch_algorithm alone is checked below.
TEST(TestRansacKabsch, RecoversTransformDespiteOutliers) {
  const Eigen::Matrix3d R = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).matrix();
  const Eigen::Vector3d t{0.5, -0.2, 0.1};

  const int n = 120, n_out = 30;      // 25% outliers, matching the measured rate
  Eigen::MatrixX3d P(n, 3);
  for (int k = 0; k < n; ++k)         // deterministic spread, no RNG in a test
    P.row(k) << std::sin(0.7 * k), std::cos(1.3 * k), 1.0 + 0.5 * std::sin(2.1 * k);
  Eigen::MatrixX3d Q = (P * R.transpose()).rowwise() + t.transpose();
  for (int k = 0; k < n_out; ++k) Q.row(k) += Eigen::Vector3d{3.0, -2.5, 4.0}.transpose();

  const auto fit = kabsch_ransac(P, Q);

  EXPECT_LT(angle_between(fit.T_ji.topLeftCorner<3, 3>(), R), 1e-9);
  EXPECT_LT((fit.T_ji.topRightCorner<3, 1>() - t).norm(), 1e-9);
  EXPECT_EQ(fit.inliers.size(), static_cast<size_t>(n - n_out));
  // no planted outlier survived
  EXPECT_TRUE(std::none_of(fit.inliers.begin(), fit.inliers.end(),
                           [n_out](Eigen::Index i) { return i < n_out; }));

  // the contrast that justifies the ticket
  const Eigen::Matrix4d naive = kabsch_algorithm(P, Q).T_ji;
  EXPECT_GT(angle_between(naive.topLeftCorner<3, 3>(), R), 1e-3);
}

// Collinear points leave rotation about the line undetermined, so EVERY sample
// is degenerate. Must signal failure rather than return an arbitrary pose --
// note NaN degeneracy would silently pass a `< threshold` guard.
TEST(TestRansacKabsch, CollinearInputReportsFailure) {
  Eigen::MatrixX3d line(20, 3);
  for (int k = 0; k < 20; ++k) line.row(k) << k, 0.0, 0.0;
  const Eigen::Matrix4d T = transform_of(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).matrix(), {0.5, -0.2, 0.1});
  const Eigen::MatrixX3d moved =
      (line * T.topLeftCorner<3, 3>().transpose()).rowwise()
      + T.topRightCorner<3, 1>().transpose();

  const auto fit = kabsch_ransac(line, moved);

  EXPECT_TRUE(fit.inliers.empty());
  EXPECT_EQ(fit.inlier_ratio, 0.0);
}
