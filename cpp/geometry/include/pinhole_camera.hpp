#pragma once
#include <Eigen/Core>

namespace geometry::camera {

  class PinholeCamera {
  public:
    PinholeCamera(const double fx, const double fy, const double cx, const double cy) : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    Eigen::Matrix3d matrix() const;
    Eigen::Vector2d project(const Eigen::Vector3d& point) const;
    Eigen::Vector3d back_project(const Eigen::Vector2d& pixel, double depth) const;

    double fx() const { return fx_; }
    double fy() const { return fy_; }
    double cx() const { return cx_; }
    double cy() const { return cy_; }

  private:
    const double fx_, fy_, cx_, cy_;
  };
}
