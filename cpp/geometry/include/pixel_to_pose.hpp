#pragma once
#include <Eigen/Dense>

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
}
