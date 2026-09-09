#include "graph_struct.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Geometry>
#include <Eigen/Sparse>


namespace pose_graph {

  FactorGraph::FactorGraph(const std::filesystem::path &path_to_g2o_file) {

    if (!std::filesystem::exists(path_to_g2o_file)) {
      throw std::runtime_error("File does not exist");
    }
    std::ifstream file(path_to_g2o_file);
    std::string line;

    std::unordered_map<int, int> id_to_index;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string type;
      iss >> type;

      if (type == "VERTEX_SE3:QUAT") {
        int id;
        double x, y, z;
        double qx, qy, qz, qw;

        iss >> id >> x >> y >> z
           >> qx >> qy >> qz >> qw;
        auto pose = geometry::PoseSE3(x, y, z, qx, qy, qz, qw);
        vertices.emplace_back(pose, 0);
        id_to_index[id] = vertices.size() - 1;
      }
      else if (type == "EDGE_SE3:QUAT") {
        int i, j;
        double tx, ty, tz;
        double qx, qy, qz, qw;

        iss >> i >> j
           >> tx >> ty >> tz
           >> qx >> qy >> qz >> qw;

        Information info;
        info.setZero();

        for (int r = 0; r < 6; ++r) {
          for (int c = r; c < 6; ++c) {
            double value;
            if (!(iss >> value)) {
              throw std::runtime_error("EDGE_SE3:QUAT needs 21 information values");
            }
            info(r, c) = value;
            info(c, r) = value;
          }
        }

        edges.emplace_back(i, j, geometry::PoseSE3(tx, ty, tz, qx, qy, qz, qw), info);
      }
    }

    for (auto& edge : edges) {
      edge.from = id_to_index.at(edge.from);
      edge.to = id_to_index.at(edge.to);
    }
  }

  int FactorGraph::add_vertex(const Eigen::Matrix4d &pose, const double timestamp) {
    vertices.emplace_back(geometry::PoseSE3(pose), timestamp);
    return static_cast<int>(vertices.size()) - 1;
  }

  void FactorGraph::add_edge(const int from, const int to,
                             const Eigen::Matrix4d &measurement,
                             const Information &information) {
    const int num_vertices = static_cast<int>(vertices.size());
    if (from < 0 || from >= num_vertices || to < 0 || to >= num_vertices) {
      throw std::invalid_argument("edge endpoint out of range");
    }
    edges.emplace_back(from, to, geometry::PoseSE3(measurement), information);
  }

  void FactorGraph::save_to_file(const std::filesystem::path &path_to_g2o_file) const {
    std::ofstream file(path_to_g2o_file);
    if (!file) {
      throw std::runtime_error("cannot open for writing: " + path_to_g2o_file.string());
    }
    // 17 significant digits is the shortest that round-trips a double exactly,
    // so save/load is lossless and a round-trip test can assert equality.
    file << std::setprecision(17);

    for (size_t index = 0; index < vertices.size(); ++index) {
      const Eigen::Vector3d t = vertices[index].pose.translation();
      const Eigen::Quaterniond q(vertices[index].pose.rotation().matrix());
      file << "VERTEX_SE3:QUAT " << index
           << ' ' << t.x() << ' ' << t.y() << ' ' << t.z()
           << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }

    for (const Edge &edge : edges) {
      const Eigen::Vector3d t = edge.measured_pose.translation();
      const Eigen::Quaterniond q(edge.measured_pose.rotation().matrix());
      file << "EDGE_SE3:QUAT " << edge.from << ' ' << edge.to
           << ' ' << t.x() << ' ' << t.y() << ' ' << t.z()
           << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w();
      for (int r = 0; r < 6; ++r) {
        for (int c = r; c < 6; ++c) file << ' ' << edge.info_matrix(r, c);
      }
      file << '\n';
    }
  }

  FactorGraph::FactorGraph(const std::vector<Eigen::Matrix4d> &poses,
                           const std::vector<double> &timestamps,
                           const std::vector<int> &edge_from,
                           const std::vector<int> &edge_to,
                           const std::vector<Eigen::Matrix4d> &measurements,
                           const std::vector<Information> &information) {
    if (poses.size() != timestamps.size()) {
      throw std::invalid_argument("poses and timestamps must have equal length");
    }
    const size_t num_edges = edge_from.size();
    if (edge_to.size() != num_edges || measurements.size() != num_edges
        || information.size() != num_edges) {
      throw std::invalid_argument(
          "edge_from, edge_to, measurements and information must have equal length");
    }

    vertices.reserve(poses.size());
    for (size_t v = 0; v < poses.size(); ++v) {
      vertices.emplace_back(geometry::PoseSE3(poses[v]), timestamps[v]);
    }

    // An out-of-range endpoint would index past the vertex array in the solver,
    // which reads garbage rather than failing -- check here, once.
    const int num_vertices = static_cast<int>(vertices.size());
    edges.reserve(num_edges);
    for (size_t e = 0; e < num_edges; ++e) {
      if (edge_from[e] < 0 || edge_from[e] >= num_vertices
          || edge_to[e] < 0 || edge_to[e] >= num_vertices) {
        throw std::invalid_argument("edge endpoint out of range");
      }
      edges.emplace_back(edge_from[e], edge_to[e],
                         geometry::PoseSE3(measurements[e]), information[e]);
    }
  }

  void FactorGraph::gauss_newton(const double iter_threshold, const bool verbose, const double huber) {
    if (vertices.size() < 2) {
      throw std::invalid_argument("gauss_newton needs at least 2 vertices");
    }
    float delta_mag = std::numeric_limits<float>::infinity();

    // Gauss-Newton can diverge, and an unbounded loop turns that into a hang
    // rather than a diagnosable result.
    constexpr int max_iterations = 100;
    int iterations = 0;

    Eigen::VectorXd residuals(6 * edges.size());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H(6*vertices.size(), 6*vertices.size());
    Eigen::VectorXd b(6 * vertices.size());

    while (delta_mag > iter_threshold && iterations++ < max_iterations) {
      H.setZero();
      b.setZero();

      for (int i = 0; i < edges.size(); i++) {
        Edge &edge = edges[i];
        geometry::PoseSE3 trans_error = edge.measured_pose.inverse() * vertices[edge.from].pose.inverse() * vertices[edge.to].pose;
        residuals.segment<6>(i * 6) = trans_error.log();

        Eigen::Matrix<double, 6, 6> J_i = -vertices[edge.to].pose.inverse().adjoint();
        Eigen::Matrix<double, 6, 6> J_j = vertices[edge.to].pose.inverse().adjoint();

        double w = 1;
        if (huber != 0.0) {
          const auto r = residuals.segment<6>(i * 6);
          const double mahalanobis = std::sqrt(r.dot(edge.info_matrix * r));
          w = std::min(1.0, huber / mahalanobis);
        }

        H.block<6,6>(edge.from*6,edge.from*6) += w * J_i.transpose() * edge.info_matrix * J_i;
        H.block<6,6>(edge.to*6,edge.from*6) += w * J_j.transpose() * edge.info_matrix * J_i;
        H.block<6,6>(edge.from*6,edge.to*6) += w * J_i.transpose() * edge.info_matrix * J_j;
        H.block<6,6>(edge.to*6,edge.to*6) += w * J_j.transpose() * edge.info_matrix * J_j;

        b.segment<6>(edge.from*6) += w * -J_i.transpose() * edge.info_matrix * residuals.segment<6>(i * 6);
        b.segment<6>(edge.to*6) += w * -J_j.transpose() * edge.info_matrix * residuals.segment<6>(i * 6);


      }
      // Gauge fix: absolute poses are defined only up to one global rigid
      // transform, so H is singular by exactly 6. Vertex 0 is the anchor -- drop
      // its rows and columns rather than solve a singular system, because LDLT
      // does not fail on one, it returns an arbitrary member of a 6-dimensional
      // family. `reduced` therefore indexes vertex k as block k-1.
      const Eigen::Index reduced = 6 * (static_cast<Eigen::Index>(vertices.size()) - 1);
      // H is assembled DENSELY above, so this makes only the factorisation
      // sparse -- the O(V^2) build and memory are still paid. Assembling from
      // triplets is the rest of that win.
      const Eigen::SparseMatrix<double> H_reduced =
          H.bottomRightCorner(reduced, reduced).sparseView();
      const Eigen::VectorXd b_reduced = b.tail(reduced);

      // SparseMatrix has no .ldlt(); SimplicialLDLT is the sparse Cholesky.
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(H_reduced);
      if (solver.info() != Eigen::Success) {
        if (verbose) std::cout << "  sparse factorisation failed -- stopping\n";
        return;
      }
      const Eigen::VectorXd delta = solver.solve(b_reduced);
      delta_mag = delta.norm();

      if (verbose) {
        double cost = 0.0;
        for (size_t e = 0; e < edges.size(); ++e) {
          const auto r = residuals.segment<6>(e * 6);
          cost += r.transpose() * edges[e].info_matrix * r;
        }
        std::cout << "  iter " << iterations << "  cost " << cost
                  << "  |delta| " << delta_mag << '\n';
      }

      // Diverging to NaN silently replaces every pose with garbage. Bail out and
      // leave the graph as it was rather than destroy the caller's trajectory.
      if (!delta.allFinite()) {
        if (verbose) std::cout << "  delta is not finite -- stopping\n";
        return;
      }

      for (size_t v = 1; v < vertices.size(); ++v) {
        // (v - 1) because vertex 0 was removed above; using v would hand each
        // vertex its neighbour's correction and read past the end of delta
        const geometry::PoseSE3 update =
            geometry::PoseSE3::exp(delta.segment<6>((v - 1) * 6));
        vertices[v].pose = update * vertices[v].pose;
      }

    }


  }
}
