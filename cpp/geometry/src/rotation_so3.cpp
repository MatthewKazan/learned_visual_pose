#include "rotation_so3.hpp"
#include "utils.hpp"

namespace geometry {

  RotationSO3::RotationSO3(const Eigen::Matrix3d& R) {
    auto identity_check = R.transpose() * R;
    if (!identity_check.isIdentity(1e-6)
      || std::abs(R.determinant() - 1.0) > 1e-9) {
      throw std::runtime_error("Given matrix is not in SO(3)");
    }
    R_ = R;
  }
  RotationSO3 RotationSO3::fromValidMatrix(const Eigen::Matrix3d& R) {
    auto r = RotationSO3();
    r.R_ = R;
    return r;
  }

  /**
   * Exponential map from lie algebra so3 to SO3.
   *
   * @param omega axis angle rotation representation
   * @return Rotation matrix in SO3
   */
  RotationSO3 RotationSO3::exp(const Eigen::Vector3d& omega) {
    const Eigen::Matrix3d omega_skew = hat(omega);
    const double theta = omega.norm();

    const Eigen::Matrix3d R =
        Eigen::Matrix3d::Identity()
        + sin_over_theta(theta) * omega_skew
        + one_minus_cos_over_theta2(theta) * omega_skew * omega_skew;

    return fromValidMatrix(R);
  }
  const Eigen::Matrix3d& RotationSO3::matrix() const {
    return R_;
  }
  Eigen::Vector3d RotationSO3::log() const {
    // clamp needed because of floating point error
    const double cos_theta =
        std::clamp((R_.trace() - 1.0) / 2.0, -1.0, 1.0);

    const double theta = std::acos(cos_theta);

    if (theta < 1e-6) {
      // b/c when theta close to 0 -> theta / 2 * sin(theta) ~= 1/2
      return vee(0.5 * (R_ - R_.transpose()));
    }

    // TODO: dont fully understand this part yet
    if (M_PI - theta < 1e-6) {
      // Near theta = pi: R is symmetric so (R - R^T) drops the axis.
      // Recover it from R + I = 2 * omega * omega^T instead.
      // Pick the largest diagonal of M as the pivot component so we
      // divide by the biggest sqrt (avoids amplifying noise).
      const Eigen::Matrix3d M = 0.5 * (R_ + Eigen::Matrix3d::Identity());

      int k = 0;
      if (M(1, 1) > M(k, k)) k = 1;
      if (M(2, 2) > M(k, k)) k = 2;
      const int i = (k + 1) % 3;
      const int j = (k + 2) % 3;

      Eigen::Vector3d axis;
      axis(k) = std::sqrt(std::max(M(k, k), 0.0));
      axis(i) = M(k, i) / axis(k);
      axis(j) = M(k, j) / axis(k);

      return theta * axis;
    }

    return vee(theta / (2.0 * std::sin(theta)) * (R_ - R_.transpose()));
  }
  RotationSO3 RotationSO3::inverse() const {
    return fromValidMatrix(R_.transpose());
  }
  RotationSO3 RotationSO3::operator*(const RotationSO3 &other) const {
    return fromValidMatrix(R_ * other.R_);
  }
  Eigen::Vector3d RotationSO3::operator*(const Eigen::Vector3d &other) const {
    return R_ * other;
  }

}