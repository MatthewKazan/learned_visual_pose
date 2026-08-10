#ifndef LEARNED_VISUAL_POSE_GEOMETRY_HPP
#define LEARNED_VISUAL_POSE_GEOMETRY_HPP

#include <Eigen/Core>

namespace geometry {

  Eigen::Matrix3d hat(const Eigen::Vector3d& w);
  Eigen::Vector3d vee(const Eigen::Matrix3d& W);

  class RotationSO3 {
  public:
    RotationSO3() : R_(Eigen::Matrix3d::Identity()) {}
    explicit RotationSO3(const Eigen::Matrix3d& R);

    const Eigen::Matrix3d& matrix() const;
    Eigen::Vector3d log() const;
    RotationSO3 inverse() const;
    RotationSO3 operator*(const RotationSO3 &other) const;
    Eigen::Vector3d operator*(const Eigen::Vector3d &other) const;

    static RotationSO3 exp(const Eigen::Vector3d &omega);

  private:
    Eigen::Matrix3d R_;
    static RotationSO3 fromValidMatrix(const Eigen::Matrix3d& R) ;
  };
}
#endif //LEARNED_VISUAL_POSE_GEOMETRY_HPP
