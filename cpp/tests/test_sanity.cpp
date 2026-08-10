#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <geometry.hpp>
#include <cmath>

const Eigen::Matrix3d validRotationMatrix1{
    {0, -1, 0},
    {1,  0, 0},
    {0,  0, 1},
};
const Eigen::Matrix3d validRotationMatrix2 = Eigen::Quaternion<double>(1, 0.1, 0.2, 0.3).normalized().toRotationMatrix();
const Eigen::Matrix3d validRotationMatrix1T{
    {0, 1, 0},
    {-1,  0, 0},
    {0,  0, 1},
};

const Eigen::Matrix3d invalidRotationMatrix1{
    {0, -1, -30},
    {1,  0, 1},
    {0,  4, 1},
};

TEST(SanityTest, ArithmeticWorks) {
    EXPECT_EQ(2 + 2, 4);
    EXPECT_EQ(validRotationMatrix1T, validRotationMatrix1.transpose());
}

TEST(GeometryTest, ConstructionPass) {
    geometry::RotationSO3 r1(validRotationMatrix1);
    geometry::RotationSO3 r2(validRotationMatrix2);
    const Eigen::Matrix3d r3 = validRotationMatrix1 * validRotationMatrix2;
    geometry::RotationSO3 r4(r3);
}

TEST(GeometryTest, ConstructionFail) {
    EXPECT_ANY_THROW(geometry::RotationSO3{invalidRotationMatrix1});
}

TEST(GeometryTest, Inverse) {
    geometry::RotationSO3 r(validRotationMatrix1);
    EXPECT_EQ(r.inverse().matrix(), validRotationMatrix1T);
    EXPECT_EQ(r.inverse().matrix(), validRotationMatrix1.inverse());
    EXPECT_EQ(r.inverse().matrix(), validRotationMatrix1.transpose());
    EXPECT_EQ(r.inverse().inverse().matrix(), validRotationMatrix1);
    auto r_i = r.inverse();
    auto r2 = r_i * r;
    EXPECT_EQ(r2.matrix(), Eigen::Matrix3d::Identity());
}

TEST(GeometryTest, DefaultIsIdentity) {
    geometry::RotationSO3 r;
    EXPECT_EQ(r.matrix(), Eigen::Matrix3d::Identity());
}

TEST(GeometryTest, MatrixAccessor) {
    geometry::RotationSO3 r(validRotationMatrix1);
    EXPECT_EQ(r.matrix(), validRotationMatrix1);
}

TEST(GeometryTest, ActsOnVector) {
    // validRotationMatrix1 = 90° rotation about z-axis: x -> y, y -> -x, z -> z
    geometry::RotationSO3 r(validRotationMatrix1);
    EXPECT_EQ(r * Eigen::Vector3d::UnitX(),  Eigen::Vector3d::UnitY());
    EXPECT_EQ(r * Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitX());
    EXPECT_EQ(r * Eigen::Vector3d::UnitZ(),  Eigen::Vector3d::UnitZ());
}

TEST(GeometryTest, Composition) {
    geometry::RotationSO3 r(validRotationMatrix1);  // 90° about z
    const Eigen::Matrix3d expected_180{
        {-1,  0, 0},
        { 0, -1, 0},
        { 0,  0, 1},
    };
    EXPECT_EQ((r * r).matrix(), expected_180);
    EXPECT_EQ((r * r * r * r).matrix(), Eigen::Matrix3d::Identity());
}

TEST(GeometryTest, ExpOfZeroIsIdentity) {
    auto r = geometry::RotationSO3::exp(Eigen::Vector3d::Zero());
    EXPECT_TRUE(r.matrix().isApprox(Eigen::Matrix3d::Identity()));
}

TEST(GeometryTest, ExpOfAxisAlignedAngle) {
    // 90° about z-axis -> validRotationMatrix1
    auto r = geometry::RotationSO3::exp(Eigen::Vector3d(0, 0, M_PI / 2));
    EXPECT_TRUE(r.matrix().isApprox(validRotationMatrix1));
}

TEST(GeometryTest, LogOfIdentityIsZero) {
    geometry::RotationSO3 r;
    EXPECT_TRUE(r.log().isZero(1e-9));
}

TEST(GeometryTest, LogOfAxisAligned) {
    geometry::RotationSO3 r(validRotationMatrix1);  // 90° about z
    EXPECT_TRUE(r.log().isApprox(Eigen::Vector3d(0, 0, M_PI / 2)));
}

TEST(GeometryTest, ExpLogRoundtrip) {
    geometry::RotationSO3 r(validRotationMatrix1);
    auto r_roundtrip = geometry::RotationSO3::exp(r.log());
    EXPECT_TRUE(r_roundtrip.matrix().isApprox(r.matrix()));
}

TEST(GeometryTest, LogExpRoundtrip) {
    // Kept in the injective range |omega| < π so log(exp(omega)) == omega
    const Eigen::Vector3d omega(0.1, 0.2, 0.3);
    auto r = geometry::RotationSO3::exp(omega);
    EXPECT_TRUE(r.log().isApprox(omega));
}

TEST(GeometryTest, LogAtPiAxisAligned) {
    // 180° rotation about z: R - R^T = 0, forces the theta ≈ π branch.
    const Eigen::Matrix3d R_pi_z{
        {-1,  0, 0},
        { 0, -1, 0},
        { 0,  0, 1},
    };
    const Eigen::Vector3d log_r = geometry::RotationSO3(R_pi_z).log();
    // Sign is ambiguous at π — both ±π·ẑ are valid representatives.
    EXPECT_TRUE(
        log_r.isApprox(Eigen::Vector3d(0, 0,  M_PI)) ||
        log_r.isApprox(Eigen::Vector3d(0, 0, -M_PI)));
    EXPECT_TRUE(
        geometry::RotationSO3::exp(log_r).matrix().isApprox(R_pi_z));
}

TEST(GeometryTest, LogAtPiOffAxis) {
    // 180° about (1,1,0)/√2 — exercises non-trivial pivot selection.
    const Eigen::Matrix3d R_pi_xy{
        {0, 1,  0},
        {1, 0,  0},
        {0, 0, -1},
    };
    const Eigen::Vector3d log_r = geometry::RotationSO3(R_pi_xy).log();
    const Eigen::Vector3d expected(M_PI / std::sqrt(2.0), M_PI / std::sqrt(2.0), 0);
    EXPECT_TRUE(log_r.isApprox(expected) || log_r.isApprox(-expected));
    EXPECT_TRUE(
        geometry::RotationSO3::exp(log_r).matrix().isApprox(R_pi_xy));
}

TEST(GeometryTest, RejectsReflection) {
    // Orthogonal (R^T R = I) but det = -1 — a proper reflection.
    // Lives in O(3) but not SO(3), so the constructor must reject it.
    const Eigen::Matrix3d reflection{
        {-1, 0, 0},
        { 0, 1, 0},
        { 0, 0, 1},
    };
    EXPECT_ANY_THROW(geometry::RotationSO3{reflection});
}

TEST(GeometryTest, ExpTinyAngleTaylorBranch) {
    // |omega| far below the 1e-6 threshold — must go through the Taylor
    // approximation branch inside exp() and still roundtrip cleanly.
    const Eigen::Vector3d omega(1e-9, -2e-9, 3e-9);
    auto r = geometry::RotationSO3::exp(omega);
    EXPECT_LT((r.log() - omega).norm(), 1e-12);
}

TEST(GeometryTest, ExpAtBoundaryAngle) {
    // |omega| = π exactly, along a non-axis-aligned direction. Roundtrip
    // may flip the sign of the recovered axis (inherent π ambiguity),
    // but the resulting rotation matrix must be unchanged.
    const Eigen::Vector3d axis = Eigen::Vector3d(1, 2, 3).normalized();
    auto r = geometry::RotationSO3::exp(M_PI * axis);
    auto r_roundtrip = geometry::RotationSO3::exp(r.log());
    EXPECT_TRUE(r_roundtrip.matrix().isApprox(r.matrix()));
}

TEST(GeometryTest, FullTurnIsIdentity) {
    // 2π rotation about any axis is identity.
    auto r = geometry::RotationSO3::exp(Eigen::Vector3d(0, 0, 2 * M_PI));
    EXPECT_TRUE(r.matrix().isApprox(Eigen::Matrix3d::Identity()));
}

TEST(GeometryTest, CompositionNonCommutative) {
    // 90° about x and 90° about y — classic non-commuting pair.
    auto Rx = geometry::RotationSO3::exp(Eigen::Vector3d(M_PI / 2, 0, 0));
    auto Ry = geometry::RotationSO3::exp(Eigen::Vector3d(0, M_PI / 2, 0));
    EXPECT_FALSE((Rx * Ry).matrix().isApprox((Ry * Rx).matrix()));
}

TEST(GeometryTest, PreservesVectorLength) {
    geometry::RotationSO3 r(validRotationMatrix2);
    const Eigen::Vector3d v(1.5, -2.3, 0.7);
    EXPECT_NEAR((r * v).norm(), v.norm(), 1e-12);
}

TEST(GeometryTest, PreservesInnerProduct) {
    geometry::RotationSO3 r(validRotationMatrix2);
    const Eigen::Vector3d u(1, 2, 3);
    const Eigen::Vector3d v(4, -5, 6);
    EXPECT_NEAR((r * u).dot(r * v), u.dot(v), 1e-12);
}

TEST(GeometryTest, HatTranspose) {
    const Eigen::Vector3d w(1, 2, 3);
    const Eigen::Matrix3d W = geometry::hat(w);
    EXPECT_TRUE(W.transpose().isApprox(-1 * W));
}

TEST(GeometryTest, HatCross) {
    const Eigen::Vector3d w(1, 3, 7);
    const Eigen::Matrix3d W = geometry::hat(w);
    const Eigen::Vector3d u(1, 2, 3);
    EXPECT_TRUE((W * u).isApprox(w.cross(u)));
}

TEST(GeometryTest, HatVeeRoundTrip) {
    const Eigen::Vector3d w(1, 2, 3);
    const Eigen::Matrix3d W = geometry::hat(w);
    const Eigen::Vector3d v = geometry::vee(W);
    EXPECT_TRUE(v.isApprox(w));
}