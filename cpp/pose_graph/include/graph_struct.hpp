#pragma once

#include "pose_se3.hpp"
#include <Eigen/Core>
#include <filesystem>
#include <vector>


namespace pose_graph {
  using Information = Eigen::Matrix<double, 6, 6>;

  struct Edge {
    /// INDICES into FactorGraph::vertices, not g2o vertex ids. g2o ids need not
    /// be contiguous or ordered, so the loader translates them once and every
    /// consumer downstream can index directly.
    int from;
    int to;
    // position of frame j viewed from frame i, or equivalently the transform from j to i.
    geometry::PoseSE3 measured_pose;
    Information info_matrix;
  };

  struct Vertex {
    geometry::PoseSE3 pose;
    double timestamp;
  };

  class FactorGraph {
    public:
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;

    FactorGraph() = default;
    explicit FactorGraph(const std::filesystem::path &path_to_g2o_file);

    /**
     * Build from parallel arrays -- the shape Python already has the data in.
     *
     * Endpoints are INDICES into `poses`, not g2o ids: this path does no
     * translation, because a caller assembling arrays in memory has no reason
     * to invent sparse labels. Every index is range-checked.
     */
    FactorGraph(const std::vector<Eigen::Matrix4d> &poses,
                const std::vector<double> &timestamps,
                const std::vector<int> &edge_from,
                const std::vector<int> &edge_to,
                const std::vector<Eigen::Matrix4d> &measurements,
                const std::vector<Information> &information);

    /// Returns the vertex INDEX -- the label edge endpoints use.
    int add_vertex(const Eigen::Matrix4d &pose, double timestamp = 0.0);

    /// Endpoints are vertex INDICES, range-checked here because the solver
    /// indexes `vertices` unchecked and would read garbage instead of failing.
    void add_edge(int from, int to, const Eigen::Matrix4d &measurement,
                  const Information &information);

    /// Writes vertex INDICES as g2o ids, matching the loader's normalisation.
    /// Timestamps are lost: g2o has no field for them.
    void save_to_file(const std::filesystem::path &path_to_g2o_file) const;

    /// @param verbose print cost, |delta| and iteration count -- a wrong
    ///                Jacobian shows up as a cost that rises or stalls, which
    ///                is invisible from the outside.
    void gauss_newton(double iter_threshold = 1e-6, bool verbose = false, double huber = 0.0);


  };

}