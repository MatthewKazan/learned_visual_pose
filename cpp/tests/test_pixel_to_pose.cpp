#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <gtest/gtest.h>
#include <geometry.hpp>
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
