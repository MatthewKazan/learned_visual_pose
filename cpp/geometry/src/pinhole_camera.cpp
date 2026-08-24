#include "pinhole_camera.hpp"

namespace geometry::camera {

  Eigen::Matrix3d PinholeCamera::matrix() const {
    return Eigen::Matrix3d{{fx_, 0, cx_}, {0, fy_, cy_}, {0, 0, 1}};
  }

  /**
   * Project real world point into pixel coords.
   *
   * Derivation from definition of similar triangles.
   * world_x / world_depth = pixel_x / camera_depth (how far film is from pinhole)
   *
   * @param point Real world point
   * @return Pixel coordinates
   */
  Eigen::Vector2d PinholeCamera::project(const Eigen::Vector3d& point) const {
    return Eigen::Vector2d{fx_ * point.x() / point.z() + cx_, fy_ * point.y() / point.z() + cy_};
  }

  /**
   * Project pixel coords into real world point.
   *
   * Derivation from definition of similar triangles.
   * pixel_x / camera_depth = world_x / world_depth (how far film is from pinhole)
   *
   * @param pixel Pixel coordinates
   * @param depth
   * @return
   */
  Eigen::Vector3d PinholeCamera::back_project(const Eigen::Vector2d& pixel, double depth) const {
    return Eigen::Vector3d{depth * (pixel.x() - cx_) / fx_, depth * (pixel.y() - cy_) / fy_, depth};
  }
}