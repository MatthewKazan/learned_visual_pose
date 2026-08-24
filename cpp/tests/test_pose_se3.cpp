#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <geometry.hpp>
#include <cmath>

constexpr Eigen::Matrix3d validRotationMatrix1{
  {0, -1, 0},
  {1,  0, 0},
  {0,  0, 1},
};

const Eigen::Vector3d validTranslationVector{1, 2, 3};

TEST(TestGeometrySE3, TestExpLogRoundTrip) {
  geometry::Tangent xi;
  xi << /* pick something with small phi so we stay injective */
        1.0, -2.0, 3.0,   // rho
        0.1, 0.2, 0.3;    // phi
  const auto T = geometry::PoseSE3::exp(xi);
  EXPECT_TRUE(T.log().isApprox(xi));
}

TEST(TestGeometrySE3, InverseRoundtrip) {
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  EXPECT_TRUE((T * T.inverse()).matrix().isApprox(Eigen::Matrix4d::Identity()));
  EXPECT_TRUE((T.inverse() * T).matrix().isApprox(Eigen::Matrix4d::Identity()));
}

TEST(TestGeometrySE3, DefaultIsIdentity) {
  const geometry::PoseSE3 T;
  EXPECT_TRUE(T.matrix().isApprox(Eigen::Matrix4d::Identity()));
}

TEST(TestGeometrySE3, MatrixAccessors) {
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  EXPECT_EQ(T.rotation().matrix(), validRotationMatrix1);
  EXPECT_EQ(T.translation(), validTranslationVector);
}

TEST(TestGeometrySE3, ConstructionFailFromInvalidMatrix4d) {
  // Bottom-left 3x3 is not in SO(3) — ctor should throw via RotationSO3.
  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  M.block<3,3>(0,0) << 0, -1, -30,
                       1,  0,   1,
                       0,  4,   1;
  EXPECT_ANY_THROW(geometry::PoseSE3{M});
}

TEST(TestGeometrySE3, ActsOnVector) {
  // R = 90° about z: (x -> y, y -> -x, z -> z), t = (1, 2, 3).
  // R*(1,0,0) + t = (0,1,0)  + (1,2,3) = (1,3,3)
  // R*(0,1,0) + t = (-1,0,0) + (1,2,3) = (0,2,3)
  // R*(0,0,1) + t = (0,0,1)  + (1,2,3) = (1,2,4)
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  EXPECT_EQ(T * Eigen::Vector3d::UnitX(), Eigen::Vector3d(1, 3, 3));
  EXPECT_EQ(T * Eigen::Vector3d::UnitY(), Eigen::Vector3d(0, 2, 3));
  EXPECT_EQ(T * Eigen::Vector3d::UnitZ(), Eigen::Vector3d(1, 2, 4));
  EXPECT_EQ(T * Eigen::Vector3d::Zero(),  validTranslationVector);
}

TEST(TestGeometrySE3, CompositionWithIdentity) {
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  const geometry::PoseSE3 I;
  EXPECT_TRUE((T * I).matrix().isApprox(T.matrix()));
  EXPECT_TRUE((I * T).matrix().isApprox(T.matrix()));
}

TEST(TestGeometrySE3, CompositionAssociative) {
  geometry::Tangent xi_a; xi_a << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  geometry::Tangent xi_b; xi_b << 1.0, 0.0, -1.0, 0.2, -0.1, 0.3;
  geometry::Tangent xi_c; xi_c << 0.0, 2.0, 1.0, -0.3, 0.4, -0.2;
  const auto A = geometry::PoseSE3::exp(xi_a);
  const auto B = geometry::PoseSE3::exp(xi_b);
  const auto C = geometry::PoseSE3::exp(xi_c);
  EXPECT_TRUE(((A * B) * C).matrix().isApprox((A * (B * C)).matrix()));
}

TEST(TestGeometrySE3, CompositionNonCommutative) {
  // 90° rotation about x with translation, and 90° about y with different translation.
  geometry::Tangent xi_a; xi_a << 1, 0, 0, M_PI / 2, 0, 0;
  geometry::Tangent xi_b; xi_b << 0, 1, 0, 0, M_PI / 2, 0;
  const auto A = geometry::PoseSE3::exp(xi_a);
  const auto B = geometry::PoseSE3::exp(xi_b);
  EXPECT_FALSE((A * B).matrix().isApprox((B * A).matrix()));
}

TEST(TestGeometrySE3, ExpOfZeroIsIdentity) {
  const auto T = geometry::PoseSE3::exp(geometry::Tangent::Zero());
  EXPECT_TRUE(T.matrix().isApprox(Eigen::Matrix4d::Identity()));
}

TEST(TestGeometrySE3, LogOfIdentityIsZero) {
  const geometry::PoseSE3 T;
  EXPECT_TRUE(T.log().isZero(1e-9));
}

TEST(TestGeometrySE3, ExpOfPureRotationTwist) {
  // Zero translation part, nonzero rotation part: should give (SO3::exp(phi), 0).
  geometry::Tangent xi;
  xi << 0, 0, 0,           // rho = 0
        0, 0, M_PI / 2;    // phi = 90° about z
  const auto T = geometry::PoseSE3::exp(xi);
  EXPECT_TRUE(T.rotation().matrix().isApprox(validRotationMatrix1));
  EXPECT_TRUE(T.translation().isZero(1e-12));
}

TEST(TestGeometrySE3, ExpOfPureTranslationTwist) {
  // Zero rotation part: V(0) = I, so translation part passes through untouched.
  geometry::Tangent xi;
  xi << 5, -3, 7,          // rho
        0, 0, 0;           // phi = 0
  const auto T = geometry::PoseSE3::exp(xi);
  EXPECT_TRUE(T.rotation().matrix().isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_EQ(T.translation(), Eigen::Vector3d(5, -3, 7));
}

TEST(TestGeometrySE3, LogOfPureRotation) {
  // (R, 0) where R != I: log should return (0, phi).
  const geometry::PoseSE3 T(validRotationMatrix1, Eigen::Vector3d::Zero());
  const geometry::Tangent xi = T.log();
  EXPECT_TRUE(xi.head<3>().isZero(1e-12));                             // rho = 0
  EXPECT_TRUE(xi.tail<3>().isApprox(Eigen::Vector3d(0, 0, M_PI / 2))); // phi = π/2 about z
}

TEST(TestGeometrySE3, LogOfPureTranslation) {
  // (I, t): log should return (t, 0). V^-1(0) = I, so rho = t.
  const geometry::PoseSE3 T(Eigen::Matrix3d::Identity(), Eigen::Vector3d(4, 5, -2));
  const geometry::Tangent xi = T.log();
  EXPECT_EQ(xi.head<3>(), Eigen::Vector3d(4, 5, -2));  // rho
  EXPECT_TRUE(xi.tail<3>().isZero(1e-12));             // phi
}

TEST(TestGeometrySE3, DoubleInverseIsSelf) {
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  EXPECT_TRUE(T.inverse().inverse().matrix().isApprox(T.matrix()));
}

TEST(TestGeometrySE3, InverseUndoesPointTransform) {
  // For any point p, T^-1 * (T * p) == p.
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  const Eigen::Vector3d p(1.5, -2.3, 0.7);
  const geometry::PoseSE3 T_inv = T.inverse();
  EXPECT_TRUE((T_inv * (T * p)).isApprox(p));
}

TEST(TestGeometrySE3, CompositionActsCorrectlyOnPoints) {
  // (T1 * T2) * p should equal T1 * (T2 * p).
  const geometry::PoseSE3 T1(validRotationMatrix1, validTranslationVector);
  const geometry::PoseSE3 T2(validRotationMatrix1.transpose(), Eigen::Vector3d(-1, 0.5, 2));
  const Eigen::Vector3d p(0.4, -1.1, 3.3);
  EXPECT_TRUE(((T1 * T2) * p).isApprox(T1 * (T2 * p)));
}

TEST(TestGeometrySE3, ExpTinyTangentTaylorBranch) {
  // Both rho and phi very small — hits the Taylor branch in V(phi).
  geometry::Tangent xi;
  xi << 1e-9, -2e-9, 3e-9,  // rho
        4e-10, -5e-10, 6e-10; // phi
  const auto T = geometry::PoseSE3::exp(xi);
  EXPECT_LT((T.log() - xi).norm(), 1e-12);
}

TEST(TestGeometrySE3, ExpAt2PiRotationIsIdentity) {
  // Pure 2π rotation twist: rotation wraps to identity, translation stays zero
  // (V(2π)·0 = 0). Overall the pose should be identity.
  geometry::Tangent xi;
  xi << 0, 0, 0,
        0, 0, 2 * M_PI;
  const auto T = geometry::PoseSE3::exp(xi);
  EXPECT_TRUE(T.matrix().isApprox(Eigen::Matrix4d::Identity()));
}

TEST(TestGeometrySE3, AdjointOfIdentityIsIdentity) {
  const geometry::PoseSE3 T;
  EXPECT_TRUE(T.adjoint().isApprox(geometry::Adjoint::Identity()));
}

TEST(TestGeometrySE3, AdjointHomomorphism) {
  // Ad(A * B) == Ad(A) * Ad(B).
  geometry::Tangent xi_a; xi_a << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  geometry::Tangent xi_b; xi_b << 1.0, 0.0, -1.0, 0.2, -0.1, 0.3;
  const auto A = geometry::PoseSE3::exp(xi_a);
  const auto B = geometry::PoseSE3::exp(xi_b);
  EXPECT_TRUE((A * B).adjoint().isApprox(A.adjoint() * B.adjoint()));
}

TEST(TestGeometrySE3, AdjointOfInverse) {
  // Ad(T^-1) == Ad(T)^-1.
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  EXPECT_TRUE(T.inverse().adjoint().isApprox(T.adjoint().inverse()));
}

TEST(TestGeometrySE3, AdjointConvertsLeftRightPerturbation) {
  // Core identity: Exp(Ad(T) * xi) * T == T * Exp(xi).
  // Small xi so we stay in the well-behaved regime.
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  geometry::Tangent xi; xi << 0.01, -0.02, 0.03, 0.04, -0.05, 0.06;
  const auto lhs = geometry::PoseSE3::exp(T.adjoint() * xi) * T;
  const auto rhs = T * geometry::PoseSE3::exp(xi);
  EXPECT_TRUE(lhs.matrix().isApprox(rhs.matrix()));
}

TEST(TestGeometrySE3, AdjointOfPureRotationIsBlockDiagonal) {
  // T = (R, 0): hat(t) = 0 kills the top-right coupling.
  // Ad(T) should be [[R, 0], [0, R]] — block diagonal, no cross terms.
  const geometry::PoseSE3 T(validRotationMatrix1, Eigen::Vector3d::Zero());
  const geometry::Adjoint Ad = T.adjoint();
  const Eigen::Matrix3d top_left     = Ad.block<3, 3>(0, 0);
  const Eigen::Matrix3d top_right    = Ad.block<3, 3>(0, 3);
  const Eigen::Matrix3d bottom_left  = Ad.block<3, 3>(3, 0);
  const Eigen::Matrix3d bottom_right = Ad.block<3, 3>(3, 3);
  EXPECT_TRUE(top_left.isApprox(validRotationMatrix1));
  EXPECT_TRUE(top_right.isZero(1e-12));
  EXPECT_TRUE(bottom_left.isZero(1e-12));
  EXPECT_TRUE(bottom_right.isApprox(validRotationMatrix1));
}

TEST(TestGeometrySE3, AdjointOfPureTranslationHasHatInOffDiagonal) {
  // T = (I, t): the top-right block is hat(t) * I = hat(t).
  // Diagonal blocks are identity.
  const geometry::PoseSE3 T(Eigen::Matrix3d::Identity(), validTranslationVector);
  const geometry::Adjoint Ad = T.adjoint();
  const Eigen::Matrix3d top_left     = Ad.block<3, 3>(0, 0);
  const Eigen::Matrix3d top_right    = Ad.block<3, 3>(0, 3);
  const Eigen::Matrix3d bottom_left  = Ad.block<3, 3>(3, 0);
  const Eigen::Matrix3d bottom_right = Ad.block<3, 3>(3, 3);
  EXPECT_TRUE(top_left.isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(top_right.isApprox(geometry::hat(validTranslationVector)));
  EXPECT_TRUE(bottom_left.isZero(1e-12));
  EXPECT_TRUE(bottom_right.isApprox(Eigen::Matrix3d::Identity()));
}

TEST(TestGeometrySE3, AdjointDeterminantIsOne) {
  // Ad(T) is a group representation of SE(3); its determinant is always 1.
  const geometry::PoseSE3 T(validRotationMatrix1, validTranslationVector);
  EXPECT_NEAR(T.adjoint().determinant(), 1.0, 1e-12);
}