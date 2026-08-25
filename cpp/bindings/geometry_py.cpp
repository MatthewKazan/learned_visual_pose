/**
 * Python bindings for the C++ geometry library -- the VIS-019 inference boundary.
 *
 * This is glue only: no maths lives here, and `geometry` never includes pybind,
 * so the library stays buildable and gtest-testable without Python.
 *
 * pybind11/eigen.h converts numpy <-> Eigen automatically. numpy is row-major
 * and Eigen defaults to column-major, so a (3,3) array is silently copied and
 * transposed on the way in. Correct, but it means these are copies, not views.
 */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "geometry.hpp"
#include "pixel_to_pose.hpp"

namespace py = pybind11;
using namespace geometry;

PYBIND11_MODULE(_geometry, m) {
  m.doc() = "geometry: SO(3), SE(3), pinhole camera";

  m.def("hat", &hat, py::arg("w"));
  m.def("vee", &vee, py::arg("W"));

  m.def("eight_point_algorithm", &camera::eight_point_algorithm, py::arg("ray_i"), py::arg("ray_j"));

  py::class_<RotationSO3>(m, "RotationSO3")
      .def(py::init<>())
      .def(py::init<const Eigen::Matrix3d&>(), py::arg("R"))
      .def_static("exp", &RotationSO3::exp, py::arg("omega"))
      .def("log", &RotationSO3::log)
      .def("matrix", &RotationSO3::matrix)
      .def("inverse", &RotationSO3::inverse)
      .def("__mul__", py::overload_cast<const RotationSO3&>(&RotationSO3::operator*, py::const_))
      .def("__mul__", py::overload_cast<const Eigen::Vector3d&>(&RotationSO3::operator*, py::const_))
      .def("__repr__", [](const RotationSO3& R) {
        const Eigen::Vector3d w = R.log();
        return "RotationSO3(log=[" + std::to_string(w.x()) + ", " + std::to_string(w.y())
             + ", " + std::to_string(w.z()) + "])";
      });

  py::class_<PoseSE3>(m, "PoseSE3")
      .def(py::init<>())
      .def(py::init<const Eigen::Matrix4d&>(), py::arg("T"))
      .def(py::init<const Eigen::Matrix3d&, const Eigen::Vector3d&>(), py::arg("R"), py::arg("t"))
      .def_static("exp", &PoseSE3::exp, py::arg("tangent"))
      .def("log", &PoseSE3::log)
      .def("matrix", &PoseSE3::matrix)
      .def("inverse", &PoseSE3::inverse)
      .def("adjoint", &PoseSE3::adjoint)
      .def("rotation", &PoseSE3::rotation)
      .def("translation", &PoseSE3::translation)
      .def("__mul__", py::overload_cast<const PoseSE3&>(&PoseSE3::operator*, py::const_))
      .def("__mul__", py::overload_cast<const Eigen::Vector3d&>(&PoseSE3::operator*, py::const_));

  py::class_<camera::PinholeCamera>(m, "PinholeCamera")
      .def(py::init<double, double, double, double>(),
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))
      // convenience only: Python holds intrinsics as a 3x3, C++ as four scalars
      .def(py::init([](const Eigen::Matrix3d& K) {
             return camera::PinholeCamera(K(0, 0), K(1, 1), K(0, 2), K(1, 2));
           }), py::arg("K"))
      .def("matrix", &camera::PinholeCamera::matrix)
      .def("project", &camera::PinholeCamera::project, py::arg("point"))
      .def("back_project", &camera::PinholeCamera::back_project, py::arg("pixel"), py::arg("depth"))
      .def_property_readonly("fx", &camera::PinholeCamera::fx)
      .def_property_readonly("fy", &camera::PinholeCamera::fy)
      .def_property_readonly("cx", &camera::PinholeCamera::cx)
      .def_property_readonly("cy", &camera::PinholeCamera::cy);
}
