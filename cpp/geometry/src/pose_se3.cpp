#include "pose_se3.hpp"
#include "utils.hpp"

namespace geometry {
  PoseSE3::PoseSE3(const Eigen::Matrix4d& M) {
    const Eigen::Matrix3d R = M.block<3,3>(0,0);
    const Eigen::Vector3d t = M.block<3,1>(0,3);
    R_ = RotationSO3(R);
    t_ = t;
  }

  Eigen::Matrix4d PoseSE3::matrix() const {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity(); //identity so last row is 0,0,0,1 by default
    M.block<3,3>(0,0) = R_.matrix();
    M.block<3,1>(0,3) = t_;
    return M;
  }

  PoseSE3 PoseSE3::exp(const Tangent& tangent) {
    const Eigen::Vector3d so3_tangent = tangent.tail(3);
    const RotationSO3 R = RotationSO3::exp(so3_tangent);
    const double theta = so3_tangent.norm();
    const auto so3_tangent_hat = hat(so3_tangent);

    const Eigen::Matrix3d V =
      Eigen::Matrix3d::Identity()
      + one_minus_cos_over_theta2(theta) * so3_tangent_hat
      + theta_minus_sin_over_theta3(theta) * so3_tangent_hat * so3_tangent_hat;
    const Eigen::Vector3d t = V * tangent.head(3);
    return PoseSE3(R, t);
  }

  Tangent PoseSE3::log() const {
    const Eigen::Vector3d phi = R_.log();
    Tangent tangent;
    tangent.tail(3) = phi;
    const auto w_hat = hat(phi);
    const double theta = phi.norm();

    const Eigen::Matrix3d V_inverse =
      Eigen::Matrix3d::Identity()
      - 0.5 * w_hat
      + one_over_theta2_minus_half_cot_half(theta) * w_hat * w_hat;
    tangent.head(3) = V_inverse * t_;
    return tangent;
  }

  PoseSE3 PoseSE3::inverse() const {
    const RotationSO3 R_inv = R_.inverse();
    return PoseSE3(R_inv, -(R_inv * t_));
  }

  // Twist convention: xi = [rho, phi] (translation first, rotation second).
  //   Ad(T) = [ R    hat(t) * R ]
  //           [ 0        R      ]
  Adjoint PoseSE3::adjoint() const {
    const Eigen::Matrix3d R = R_.matrix();
    Adjoint Ad = Adjoint::Zero();
    Ad.block<3, 3>(0, 0) = R;
    Ad.block<3, 3>(0, 3) = hat(t_) * R;
    Ad.block<3, 3>(3, 3) = R;
    return Ad;
  }

  PoseSE3 PoseSE3::operator*(const PoseSE3& other) const {
    return PoseSE3(R_ * other.R_, R_ * other.t_ + t_);
  }

  Eigen::Vector3d PoseSE3::operator*(const Eigen::Vector3d& other) const {
    return R_ * other + t_;
  }


}
