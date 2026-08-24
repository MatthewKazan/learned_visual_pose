#include <Eigen/Core>
#include <gtest/gtest.h>
#include <geometry.hpp>

// Convention: +X right, +Y down, +Z into the scene.
// Projection: u = fx · X/Z + cx,  v = fy · Y/Z + cy
const geometry::camera::PinholeCamera camera(500.0, 500.0, 320.0, 240.0);

TEST(TestGeometryPinholeCamera, OnAxisPointProjectsToPrincipalPoint) {
    // X = Y = 0 kills the perspective divide; only cx, cy contribute.
    EXPECT_TRUE(camera.project(Eigen::Vector3d(0, 0, 5))
                    .isApprox(Eigen::Vector2d(320, 240)));
}

TEST(TestGeometryPinholeCamera, DepthDoesNotAffectOnAxisPoints) {
    // Any depth along the optical axis maps to the same pixel.
    EXPECT_TRUE(camera.project(Eigen::Vector3d(0, 0, 100))
                    .isApprox(Eigen::Vector2d(320, 240)));
}

TEST(TestGeometryPinholeCamera, ProjectsOneMeterRight) {
    // u = 500 · 1/5 + 320 = 420
    EXPECT_TRUE(camera.project(Eigen::Vector3d(1, 0, 5))
                    .isApprox(Eigen::Vector2d(420, 240)));
}

TEST(TestGeometryPinholeCamera, TwiceTheDepthHalfTheOffset) {
    // Same X as above but Z doubled: the pixel offset from cx halves. Perspective.
    EXPECT_TRUE(camera.project(Eigen::Vector3d(1, 0, 10))
                    .isApprox(Eigen::Vector2d(370, 240)));
}

TEST(TestGeometryPinholeCamera, PointAboveCameraMapsToUpperImage) {
    // +Y is DOWN, so "above" is negative Y. v = 500·(-2)/5 + 240 = 40 — near top.
    EXPECT_TRUE(camera.project(Eigen::Vector3d(0, -2, 5))
                    .isApprox(Eigen::Vector2d(320, 40)));
}

TEST(TestGeometryPinholeCamera, BackProjectPrincipalPointGoesAlongOpticalAxis) {
    // The principal point back-projects to (0, 0, Z) for any depth.
    EXPECT_TRUE(camera.back_project(Eigen::Vector2d(320, 240), 7.0)
                    .isApprox(Eigen::Vector3d(0, 0, 7)));
}

TEST(TestGeometryPinholeCamera, ProjectBackProjectRoundtrip) {
    // project ∘ back_project must be identity on pixels for any depth > 0.
    const Eigen::Vector2d pixel(123.4, 456.7);
    const double depth = 3.5;
    EXPECT_TRUE(camera.project(camera.back_project(pixel, depth)).isApprox(pixel));
}

TEST(TestGeometryPinholeCamera, OffImagePointStillProjects) {
    // Nothing clamps pixel coords to [0, width]. Points outside the FOV project
    // to pixels outside the image; downstream code decides what to do with them.
    // u = 500·(-2)/1 + 320 = -680.
    EXPECT_TRUE(camera.project(Eigen::Vector3d(-2, 0, 1))
                    .isApprox(Eigen::Vector2d(-680, 240)));
}
