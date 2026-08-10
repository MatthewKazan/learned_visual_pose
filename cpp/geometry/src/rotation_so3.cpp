#include "rotation_so3.hpp"

#include <Eigen/LU>

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
    // construct skew symmetric metrix from axis angle omega
    Eigen::Matrix3d omega_skew;
    omega_skew << 0.0,       -omega.z(),  omega.y(),
                  omega.z(),  0.0,       -omega.x(),
                  -omega.y(), omega.x(),  0.0;

    const double theta = omega.norm();
    const double theta2 = theta * theta;

    double sin_over_theta;
    double one_minus_cos_over_theta2;

    if (theta > 1e-6) {
      sin_over_theta = std::sin(theta) / theta;
      one_minus_cos_over_theta2 =
          (1.0 - std::cos(theta)) / theta2;
    } else {
      // use taylor expansion when close to 0
      sin_over_theta = 1.0 - theta2 / 6.0;
      one_minus_cos_over_theta2 =
          0.5 - theta2 / 24.0;
    }

    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sin_over_theta * omega_skew + one_minus_cos_over_theta2 * omega_skew * omega_skew;

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
      const Eigen::Matrix3d omega_skew = 0.5 * (R_ - R_.transpose());
      return Eigen::Vector3d(
          omega_skew(2, 1),
          omega_skew(0, 2),
          omega_skew(1, 0)
      );
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

      // Sign of axis is inherently ambiguous at theta = pi
      // (pi * omega and -pi * omega represent the same rotation).
      return theta * axis;
    }

    const Eigen::Matrix3d omega_skew =
        theta / (2.0 * std::sin(theta))
        * (R_ - R_.transpose());

    return Eigen::Vector3d(
        omega_skew(2, 1),
        omega_skew(0, 2),
        omega_skew(1, 0)
    );
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