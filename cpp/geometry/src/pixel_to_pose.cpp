#include "pixel_to_pose.hpp"

#include "pinhole_camera.hpp"

namespace geometry::camera {
  Eigen::Matrix3d eight_point_algorithm(const Eigen::MatrixX3d &ray_i, const Eigen::MatrixX3d &ray_j) {
    assert(ray_i.rows() == ray_j.rows());
    assert(ray_i.rows() >= 8);
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
}
