#pragma once

#include <Eigen/Core>
#include <cmath>

namespace geometry {

  inline Eigen::Matrix3d hat(const Eigen::Vector3d& w) {
    Eigen::Matrix3d W;
    W <<  0.0,   -w.z(),  w.y(),
          w.z(),  0.0,   -w.x(),
         -w.y(),  w.x(),  0.0;
    return W;
  }

  inline Eigen::Vector3d vee(const Eigen::Matrix3d& W) {
    return Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0));
  }

  inline double sin_over_theta(double theta) {
    if (std::abs(theta) > 1e-6) {
      return std::sin(theta) / theta;
    }
    return 1.0 - theta * theta / 6.0;
  }

  inline double one_minus_cos_over_theta2(double theta) {
    if (std::abs(theta) > 1e-6) {
      return (1.0 - std::cos(theta)) / (theta * theta);
    }
    return 0.5 - theta * theta / 24.0;
  }

  inline double theta_minus_sin_over_theta3(double theta) {
    if (std::abs(theta) > 1e-6) {
      return (theta - std::sin(theta)) / std::pow(theta, 3);
    }
    // first two taylor expansion terms cancel to 1/6th, could add third taylor expansion term but unnecessary
    return 1.0 / 6.0;
  }

  /**
   * From V inverse closed form: 1/theta^2(1 - theta/2 * cot(theta/2))
   *
   * @param theta angle
   * @return result of 1/theta^2(1 - theta/2 * cot(theta/2))
   */
  inline double one_over_theta2_minus_half_cot_half(double theta) {
    if (std::abs(theta) > 1e-3) { // different threshold because math
      return (1.0 - 0.5 * theta / std::tan(0.5 * theta)) / (theta * theta);
    }
    return 1.0 / 12.0 + theta * theta / 720.0;
  }
}
