#pragma once
#include "rotation_so3.hpp"
#include <Eigen/Core>

namespace geometry {

  using Tangent = Eigen::Matrix<double, 6, 1>;
  using Adjoint = Eigen::Matrix<double, 6, 6>;

  class PoseSE3 {
  public:
    PoseSE3() : R_(RotationSO3()), t_(Eigen::Vector3d::Zero()) {}
    PoseSE3(const RotationSO3& R, const Eigen::Vector3d& t) : R_(R), t_(t) {}
    PoseSE3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) : R_(RotationSO3(R)), t_(t) {}
    PoseSE3(const double x, const double y, const double z, const double q_x, const double q_y, const double q_z, const double q_w);
    explicit PoseSE3(const Eigen::Matrix4d& M);

    const Eigen::Vector3d& translation() const { return t_; }
    const RotationSO3& rotation() const { return R_; }
    Eigen::Matrix4d matrix() const;

    Tangent log() const;
    static PoseSE3 exp(const Tangent& tangent);
    PoseSE3 inverse() const;
    // Get Matrix that changes a 6d motion vector from 1 coord frame to another
    Adjoint adjoint() const;
    PoseSE3 operator*(const PoseSE3& other) const;
    Eigen::Vector3d operator*(const Eigen::Vector3d& other) const;

  private:
    RotationSO3 R_;
    Eigen::Vector3d t_;

  };
}