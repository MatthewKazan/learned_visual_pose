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

  void FactorGraph::gauss_newton(const double iter_threshold, const bool verbose, const double huber, const double lm_lambda_init, const int max_rejections) {
    if (vertices.size() < 2) {
      throw std::invalid_argument("gauss_newton needs at least 2 vertices");
    }
    // Gauss-Newton can diverge, and an unbounded loop turns that into a hang
    // rather than a diagnosable result. Rejected LM steps count too, so a
    // stalled run ends after ~10 rejections of one step, not never.
    constexpr int max_iterations = 100;
    int iterations = 0;

    // Levenberg-Marquardt when lm_lambda_init > 0: a step is judged at the
    // start of the next iteration, where its residuals are computed anyway. A
    // step that raised the cost is undone from `saved`, lambda grows and the
    // same linearisation point is retried with a shorter step; a step that
    // lowered it shrinks lambda. lambda = 0 is plain Gauss-Newton.
    const bool damped = lm_lambda_init > 0.0;
    double lambda = lm_lambda_init;
    double prev_cost = std::numeric_limits<double>::infinity();
    double delta_mag = std::numeric_limits<double>::infinity();   // of the last ACCEPTED step
    double pending_delta_mag = std::numeric_limits<double>::infinity();   // of the last PENDING step
    int consecutive_rejections = 0;

    std::vector<geometry::PoseSE3> saved(vertices.size());
    bool pending = false;                                          // a step is applied but not yet judged

    Eigen::VectorXd residuals(6 * edges.size());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H(6*vertices.size(), 6*vertices.size());
    Eigen::VectorXd b(6 * vertices.size());

    // Residuals at the current poses, and the cost the normal equations below
    // minimise: Huber's rho(m) = m^2 for m <= k, 2km - k^2 beyond, m the
    // Mahalanobis distance. Judging on plain m^2 instead rejected good robust
    // steps, which deliberately let an outlier's m grow.
    const auto evaluate = [&]() {
      double cost = 0.0;
      for (size_t e = 0; e < edges.size(); ++e) {
        const Edge &edge = edges[e];
        const geometry::PoseSE3 trans_error = edge.measured_pose.inverse() * vertices[edge.from].pose.inverse() * vertices[edge.to].pose;
        residuals.segment<6>(e * 6) = trans_error.log();
        const auto r = residuals.segment<6>(e * 6);
        const double m2 = r.dot(edge.info_matrix * r);
        if (huber == 0.0 || m2 <= huber * huber)
          cost += m2;
        else
          cost += 2.0 * huber * std::sqrt(m2) - huber * huber;
      }
      return cost;
    };

    while (delta_mag > iter_threshold && iterations++ < max_iterations && consecutive_rejections < max_rejections) {
      const double cost = evaluate();

      // only a cost that follows an applied step is judged: the restored poses
      // after a rejection score exactly prev_cost and are not a step
      if (damped && pending) {
        pending = false;
        if (cost > prev_cost) {
          consecutive_rejections++;
          for (size_t v = 0; v < vertices.size(); ++v) vertices[v].pose = saved[v];
          lambda *= 10.0;
          delta_mag = std::numeric_limits<double>::infinity();   // a rejected step must not end the loop
          if (verbose) std::cout << "  iter " << iterations << "  cost " << cost << " > " << prev_cost << "  rejected, lambda " << lambda << '\n';
          iterations--;
          continue;                                              // re-evaluates at the restored poses
        }
        lambda = std::max(lambda / 3.0, 1e-9);
        prev_cost = cost;
        delta_mag = pending_delta_mag;
        consecutive_rejections = 0;
      }
      if (!damped) delta_mag = pending_delta_mag;

      if (verbose) {
        std::cout << "  iter " << iterations << "  cost " << cost << "  |delta| " << delta_mag;
        if (damped) std::cout << "  lambda " << lambda;
        std::cout << '\n';
      }

      H.setZero();
      b.setZero();
      for (size_t i = 0; i < edges.size(); i++) {
        Edge &edge = edges[i];
        const auto r = residuals.segment<6>(i * 6);

        Eigen::Matrix<double, 6, 6> J_i = geometry::PoseSE3::exp(r).right_jacobian().inverse() * -vertices[edge.to].pose.inverse().adjoint();
        Eigen::Matrix<double, 6, 6> J_j = geometry::PoseSE3::exp(r).right_jacobian().inverse() * vertices[edge.to].pose.inverse().adjoint();

        double w = 1;
        if (huber != 0.0) {
          const double mahalanobis = std::sqrt(r.dot(edge.info_matrix * r));
          w = std::min(1.0, huber / mahalanobis);
        }

        H.block<6,6>(edge.from*6,edge.from*6) += w * J_i.transpose() * edge.info_matrix * J_i;
        H.block<6,6>(edge.to*6,edge.from*6) += w * J_j.transpose() * edge.info_matrix * J_i;
        H.block<6,6>(edge.from*6,edge.to*6) += w * J_i.transpose() * edge.info_matrix * J_j;
        H.block<6,6>(edge.to*6,edge.to*6) += w * J_j.transpose() * edge.info_matrix * J_j;

        b.segment<6>(edge.from*6) += w * -J_i.transpose() * edge.info_matrix * r;
        b.segment<6>(edge.to*6) += w * -J_j.transpose() * edge.info_matrix * r;
      }
      // Marquardt damping: scale the diagonal so the step shortens most in the
      // directions the data constrains least. The floor keeps a direction with
      // ~zero curvature (a stretch held only by near-zero information) from
      // getting an unbounded step; with lambda = 0 this line is a no-op.
      H.diagonal() += lambda * (H.diagonal().array() + 1e-6).matrix();

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

      // Diverging to NaN silently replaces every pose with garbage. Bail out and
      // leave the graph as it was rather than destroy the caller's trajectory.
      if (!delta.allFinite()) {
        if (verbose) std::cout << "  delta is not finite -- stopping\n";
        return;
      }

      for (size_t v = 0; v < vertices.size(); ++v) saved[v] = vertices[v].pose;
      for (size_t v = 1; v < vertices.size(); ++v) {
        // (v - 1) because vertex 0 was removed above; using v would hand each
        // vertex its neighbour's correction and read past the end of delta
        const geometry::PoseSE3 update =
            geometry::PoseSE3::exp(delta.segment<6>((v - 1) * 6));
        vertices[v].pose = update * vertices[v].pose;
      }
      pending_delta_mag = delta.norm();
      pending = true;
    }

    // The loop can end with a step applied but never judged.
    if (damped && pending && evaluate() > prev_cost) {
      for (size_t v = 0; v < vertices.size(); ++v) vertices[v].pose = saved[v];
      if (verbose) std::cout << "  final step rejected\n";
    }
  }
}
