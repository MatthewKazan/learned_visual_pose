/**
 * Python bindings for the C++ geometry library.
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
#include <pybind11/stl.h>   // RansacFit::inliers is a std::vector
#include <pybind11/stl/filesystem.h>   // std::filesystem::path <-> str/Path

#include "geometry.hpp"
#include "pixel_to_pose.hpp"
#include "graph_struct.hpp"

namespace py = pybind11;
using namespace geometry;

PYBIND11_MODULE(_geometry, m) {
  m.doc() = "geometry: SO(3), SE(3), pinhole camera";

  m.def("hat", &hat, py::arg("w"));
  m.def("vee", &vee, py::arg("W"));

  m.def("eight_point_algorithm", &camera::eight_point_algorithm, py::arg("ray_i"), py::arg("ray_j"));

  m.def("set_seed", &camera::set_seed, py::arg("seed"));
  m.def("farthest_point_sample", &camera::farthest_point_sample, py::arg("points"), py::arg("num_points_to_sample"));


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

  // --- pose graph -------------------------------------------------
  py::class_<camera::KabschFit>(m, "KabschFit")
    .def_readonly("T_ji", &camera::KabschFit::T_ji)
    .def_readonly("degeneracy", &camera::KabschFit::degeneracy);

  // Constructible from Python so estimators that do not run RANSAC return this
  // same type rather than a parallel Python class.
  py::class_<camera::RansacFit>(m, "RansacFit")
    .def(py::init([](const PoseSE3 &T_ji, std::vector<Eigen::Index> inliers,
                     const double inlier_ratio) {
      return camera::RansacFit{T_ji, std::move(inliers), inlier_ratio};
    }), py::arg("T_ji"), py::arg("inliers"), py::arg("inlier_ratio"))
    .def_readonly("T_ji", &camera::RansacFit::T_ji)
    .def_readonly("inliers", &camera::RansacFit::inliers)
    .def_readonly("inlier_ratio", &camera::RansacFit::inlier_ratio)
    .def("__repr__", [](const camera::RansacFit &f) {
      return "RansacFit(" + std::to_string(f.inliers.size()) + " inliers, ratio "
           + std::to_string(f.inlier_ratio) + ")";
    });

  m.def("kabsch_algorithm", &camera::kabsch_algorithm, py::arg("point_i"), py::arg("point_j"));
  m.def("kabsch_ransac", &camera::kabsch_ransac, py::arg("point_i"), py::arg("point_j"),
        py::arg("inlier_threshold") = 0.05, py::arg("degeneracy_threshold") = 1e-3);

  py::class_<pose_graph::Vertex>(m, "Vertex")
      .def_readonly("pose", &pose_graph::Vertex::pose)
      .def_readonly("timestamp", &pose_graph::Vertex::timestamp);

  py::class_<pose_graph::Edge>(m, "Edge")
      .def_readonly("from_index", &pose_graph::Edge::from)
      .def_readonly("to_index", &pose_graph::Edge::to)
      .def_readonly("measured_pose", &pose_graph::Edge::measured_pose)
      .def_readonly("info_matrix", &pose_graph::Edge::info_matrix);

  py::class_<pose_graph::FactorGraph>(m, "FactorGraph")
      .def(py::init<>())
      .def(py::init<const std::filesystem::path &>(), py::arg("path_to_g2o_file"))
      .def(py::init<const std::vector<Eigen::Matrix4d> &, const std::vector<double> &,
                    const std::vector<int> &, const std::vector<int> &,
                    const std::vector<Eigen::Matrix4d> &,
                    const std::vector<pose_graph::Information> &>(),
           py::arg("poses"), py::arg("timestamps"), py::arg("edge_from"),
           py::arg("edge_to"), py::arg("measurements"), py::arg("information"))
      .def("add_vertex", &pose_graph::FactorGraph::add_vertex,
           py::arg("pose"), py::arg("timestamp") = 0.0,
           "Append a vertex, returning the INDEX that edge endpoints use.")
      .def("add_edge", &pose_graph::FactorGraph::add_edge,
           py::arg("from_index"), py::arg("to_index"),
           py::arg("measurement"), py::arg("information"))
      .def("save_to_file", &pose_graph::FactorGraph::save_to_file,
           py::arg("path_to_g2o_file"))
      .def("gauss_newton", &pose_graph::FactorGraph::gauss_newton,
           py::arg("iter_threshold") = 1e-6f, py::arg("verbose") = false, py::arg("huber") = false)
      // (N, 4, 4) of the current estimates -- what the caller wants back after
      // gauss_newton, without walking `vertices` element by element in Python
      .def("poses", [](const pose_graph::FactorGraph &g) {
        std::vector<Eigen::Matrix4d> out;
        out.reserve(g.vertices.size());
        for (const pose_graph::Vertex &v : g.vertices) out.push_back(v.pose.matrix());
        return out;
      })
      // The write half of poses(): a non-C++ solver produces poses rather than
      // mutating vertices, and `vertices` is a read-only copy.
      .def("set_poses", [](pose_graph::FactorGraph &g,
                           const std::vector<Eigen::Matrix4d> &poses) {
        if (poses.size() != g.vertices.size()) {
          throw std::invalid_argument("set_poses needs one pose per vertex");
        }
        for (size_t v = 0; v < poses.size(); ++v) {
          g.vertices[v].pose = geometry::PoseSE3(poses[v]);
        }
      }, py::arg("poses"))
      // `info_matrix` is def_readonly, so Python cannot rescale an edge after the
      // fact; normalising information across the graph needs every edge to exist first.
      .def("set_information", [](pose_graph::FactorGraph &g,
                                 const std::vector<pose_graph::Information> &information) {
        if (information.size() != g.edges.size()) {
          throw std::invalid_argument("set_information needs one information matrix per edge");
        }
        for (size_t e = 0; e < information.size(); ++e) {
          g.edges[e].info_matrix = information[e];
        }
      }, py::arg("information"))
      .def_readonly("vertices", &pose_graph::FactorGraph::vertices)
      .def_readonly("edges", &pose_graph::FactorGraph::edges)
      .def("__len__", [](const pose_graph::FactorGraph &g) { return g.vertices.size(); })
      .def("__repr__", [](const pose_graph::FactorGraph &g) {
        return "FactorGraph(" + std::to_string(g.vertices.size()) + " vertices, "
             + std::to_string(g.edges.size()) + " edges)";
      });

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
    // Batch overload: the frontend holds (N, 2) pixels and (N,) depths.
    // Declared second so a (2,) + scalar call still binds to the one above.
    .def("back_project", [](const camera::PinholeCamera &cam,
                            const Eigen::Ref<const Eigen::MatrixX2d> &pixels,
                            const Eigen::Ref<const Eigen::VectorXd> &depth) {
      if (pixels.rows() != depth.size()) {
        throw std::invalid_argument("pixels and depth must have equal length");
      }
      Eigen::MatrixX3d out(pixels.rows(), 3);
      for (Eigen::Index n = 0; n < pixels.rows(); ++n) {
        out.row(n) = cam.back_project(Eigen::Vector2d(pixels(n, 0), pixels(n, 1)), depth(n));
      }
      return out;
    }, py::arg("pixels"), py::arg("depth"))
    .def_property_readonly("fx", &camera::PinholeCamera::fx)
    .def_property_readonly("fy", &camera::PinholeCamera::fy)
    .def_property_readonly("cx", &camera::PinholeCamera::cx)
    .def_property_readonly("cy", &camera::PinholeCamera::cy);
}
