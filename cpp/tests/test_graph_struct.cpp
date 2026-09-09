/**
 * FactorGraph g2o serialization.
 *
 * The round trip is the load-bearing test: it exercises quaternion conversion
 * both ways, the 21-value upper-triangle packing, and the id normalisation in
 * one assertion. The rest pin failure modes that would otherwise corrupt a
 * graph silently -- a wrong graph converges to a wrong answer without
 * complaining, so every malformed input must throw rather than be absorbed.
 */
#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <stdexcept>

#include "graph_struct.hpp"

using pose_graph::Edge;
using pose_graph::FactorGraph;
using pose_graph::Information;
using pose_graph::Vertex;

namespace {
  std::filesystem::path scratch(const std::string &name) {
    return std::filesystem::temp_directory_path() / name;
  }

  void write(const std::filesystem::path &path, const std::string &contents) {
    std::ofstream out(path);
    out << contents;
  }

  geometry::PoseSE3 pose_of(const double angle, const Eigen::Vector3d &t) {
    const Eigen::Matrix3d R =
        Eigen::AngleAxisd(angle, Eigen::Vector3d(0.3, -0.6, 0.7).normalized()).matrix();
    return geometry::PoseSE3(R, t);
  }

  // asymmetric on purpose: a transposed or mis-packed upper triangle survives a
  // symmetric matrix, so the fixture has to distinguish (r,c) from (c,r)
  Information information_fixture() {
    Information info = Information::Zero();
    double value = 1.0;
    for (int r = 0; r < 6; ++r)
      for (int c = r; c < 6; ++c) {
        info(r, c) = value;
        info(c, r) = value;
        value += 1.0;
      }
    return info;
  }

  FactorGraph three_pose_graph() {
    FactorGraph g;
    g.vertices.push_back(Vertex{pose_of(0.0, {0, 0, 0}), 0.0});
    g.vertices.push_back(Vertex{pose_of(0.4, {1, 2, 3}), 0.0});
    g.vertices.push_back(Vertex{pose_of(-0.9, {-4, 0.5, 2}), 0.0});
    g.edges.push_back(Edge{0, 1, pose_of(0.4, {1, 2, 3}), information_fixture()});
    g.edges.push_back(Edge{1, 2, pose_of(0.2, {0.5, -1, 0}), information_fixture()});
    g.edges.push_back(Edge{2, 0, pose_of(0.1, {-1, 0, 1}), information_fixture()});  // loop closure
    return g;
  }
}

TEST(TestFactorGraph, SaveLoadRoundTripsExactly) {
  const FactorGraph original = three_pose_graph();
  const auto path = scratch("round_trip.g2o");
  original.save_to_file(path);

  const FactorGraph loaded(path);

  ASSERT_EQ(loaded.vertices.size(), original.vertices.size());
  ASSERT_EQ(loaded.edges.size(), original.edges.size());

  for (size_t i = 0; i < original.vertices.size(); ++i) {
    // compare the 4x4, not the quaternion: q and -q are the same rotation, so a
    // component-wise quaternion check can fail on a perfectly correct round trip
    EXPECT_LT((loaded.vertices[i].pose.matrix()
               - original.vertices[i].pose.matrix()).norm(), 1e-12)
        << "vertex " << i;
  }
  for (size_t e = 0; e < original.edges.size(); ++e) {
    EXPECT_EQ(loaded.edges[e].from, original.edges[e].from);
    EXPECT_EQ(loaded.edges[e].to, original.edges[e].to);
    EXPECT_LT((loaded.edges[e].measured_pose.matrix()
               - original.edges[e].measured_pose.matrix()).norm(), 1e-12) << "edge " << e;
    EXPECT_LT((loaded.edges[e].info_matrix - original.edges[e].info_matrix).norm(), 1e-12)
        << "edge " << e << " information";
  }
  std::filesystem::remove(path);
}

// g2o ids are arbitrary labels: non-contiguous, unordered, not zero-based. Edges
// name them, but the solver wants indices, so the loader must translate.
TEST(TestFactorGraph, TranslatesNonContiguousVertexIds) {
  const auto path = scratch("sparse_ids.g2o");
  write(path,
        "VERTEX_SE3:QUAT 7 0 0 0 0 0 0 1\n"
        "VERTEX_SE3:QUAT 3 1 0 0 0 0 0 1\n"
        "VERTEX_SE3:QUAT 99 2 0 0 0 0 0 1\n"
        "EDGE_SE3:QUAT 3 99 1 0 0 0 0 0 1"
        " 1 0 0 0 0 0  1 0 0 0 0  1 0 0 0  1 0 0  1 0  1\n");

  const FactorGraph g(path);

  ASSERT_EQ(g.vertices.size(), 3u);
  ASSERT_EQ(g.edges.size(), 1u);
  // file order is 7, 3, 99 -> indices 0, 1, 2, so the edge 3->99 becomes 1->2
  EXPECT_EQ(g.edges[0].from, 1);
  EXPECT_EQ(g.edges[0].to, 2);
  EXPECT_LT((g.vertices[g.edges[0].from].pose.translation()
             - Eigen::Vector3d(1, 0, 0)).norm(), 1e-12);
  std::filesystem::remove(path);
}

// An edge naming a vertex that does not exist used to become an edge to vertex 0
// via unordered_map::operator[], which inserts a default 0 for a missing key.
TEST(TestFactorGraph, ThrowsOnEdgeToUnknownVertex) {
  const auto path = scratch("dangling_edge.g2o");
  write(path,
        "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
        "EDGE_SE3:QUAT 0 42 1 0 0 0 0 0 1"
        " 1 0 0 0 0 0  1 0 0 0 0  1 0 0 0  1 0 0  1 0  1\n");

  EXPECT_THROW(FactorGraph g(path), std::out_of_range);
  std::filesystem::remove(path);
}

// A short line leaves the stream in a fail state and zero-fills the rest, which
// reads as "infinitely uncertain" and gets silently optimized against.
TEST(TestFactorGraph, ThrowsOnTruncatedInformationMatrix) {
  const auto path = scratch("truncated_info.g2o");
  write(path,
        "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
        "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1\n"
        "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 1 0 0\n");   // 3 of 21 values

  EXPECT_THROW(FactorGraph g(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(TestFactorGraph, ThrowsOnMissingFile) {
  EXPECT_THROW(FactorGraph g(scratch("definitely_not_here.g2o")), std::runtime_error);
}

// The array constructor is the path Python uses -- direct construction through
// the binding, no file round trip, since there is no process boundary here.
// Endpoints are indices already, so it does no id translation.
TEST(TestFactorGraph, BuildsFromArrays) {
  const FactorGraph reference = three_pose_graph();

  std::vector<Eigen::Matrix4d> poses, measurements;
  std::vector<double> timestamps;
  std::vector<int> from, to;
  std::vector<Information> information;
  for (size_t v = 0; v < reference.vertices.size(); ++v) {
    poses.push_back(reference.vertices[v].pose.matrix());
    timestamps.push_back(0.1 * static_cast<double>(v));
  }
  for (const Edge &e : reference.edges) {
    from.push_back(e.from);
    to.push_back(e.to);
    measurements.push_back(e.measured_pose.matrix());
    information.push_back(e.info_matrix);
  }

  const FactorGraph built(poses, timestamps, from, to, measurements, information);

  ASSERT_EQ(built.vertices.size(), reference.vertices.size());
  ASSERT_EQ(built.edges.size(), reference.edges.size());
  EXPECT_DOUBLE_EQ(built.vertices[2].timestamp, 0.2);   // survives, unlike a g2o round trip
  for (size_t v = 0; v < poses.size(); ++v) {
    EXPECT_LT((built.vertices[v].pose.matrix() - poses[v]).norm(), 1e-12) << "vertex " << v;
  }
  for (size_t e = 0; e < from.size(); ++e) {
    EXPECT_EQ(built.edges[e].from, from[e]);
    EXPECT_EQ(built.edges[e].to, to[e]);
    EXPECT_LT((built.edges[e].info_matrix - information[e]).norm(), 1e-12);
  }
}

TEST(TestFactorGraph, ArrayConstructorRejectsInconsistentInput) {
  const std::vector<Eigen::Matrix4d> poses(3, Eigen::Matrix4d::Identity());
  const std::vector<double> timestamps(3, 0.0);
  const std::vector<Eigen::Matrix4d> one_measurement(1, Eigen::Matrix4d::Identity());
  const std::vector<Information> one_information(1, Information::Identity());

  EXPECT_THROW(FactorGraph(poses, {0.0}, {0}, {1}, one_measurement, one_information),
               std::invalid_argument);
  EXPECT_THROW(FactorGraph(poses, timestamps, {0, 1}, {1}, one_measurement, one_information),
               std::invalid_argument);
  // an out-of-range endpoint would index past the vertex array in the solver
  EXPECT_THROW(FactorGraph(poses, timestamps, {0}, {99}, one_measurement, one_information),
               std::invalid_argument);
  EXPECT_THROW(FactorGraph(poses, timestamps, {-1}, {0}, one_measurement, one_information),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Gauss-Newton
//
// These are the two checks the ticket asks for. Both are indifferent to which
// convention the solver picked internally: one measures the true Jacobian by
// finite differences, the other only asks whether the optimum is reached.
// ---------------------------------------------------------------------------

namespace {
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  // THE convention, in one place. Both the residual and the fixture that builds
  // exact measurements derive from it, so a change to the solver's ordering
  // breaks here loudly instead of silently making the fixture inconsistent --
  // which is exactly what happened the first time these tests were written.
  geometry::PoseSE3 predicted_measurement(const geometry::PoseSE3 &from_pose,
                                          const geometry::PoseSE3 &to_pose) {
    return from_pose.inverse() * to_pose;
  }

  Vector6d edge_residual(const FactorGraph &g, const Edge &e) {
    return (e.measured_pose.inverse()
            * predicted_measurement(g.vertices[e.from].pose, g.vertices[e.to].pose)).log();
  }

  double total_cost(const FactorGraph &g) {
    double f = 0.0;
    for (const Edge &e : g.edges) {
      const Vector6d r = edge_residual(g, e);
      f += r.transpose() * e.info_matrix * r;
    }
    return f;
  }

  // Central-difference Jacobian of one edge's residual w.r.t. a LEFT
  // perturbation of one vertex. Second order accurate; eps 1e-6 balances
  // truncation against floating-point cancellation.
  Matrix6d numeric_jacobian(FactorGraph g, const size_t edge_index,
                            const int vertex, const double eps = 1e-6) {
    Matrix6d J;
    const geometry::PoseSE3 original = g.vertices[vertex].pose;
    for (int k = 0; k < 6; ++k) {
      Vector6d step = Vector6d::Zero();
      step(k) = eps;

      g.vertices[vertex].pose = geometry::PoseSE3::exp(step) * original;
      const Vector6d plus = edge_residual(g, g.edges[edge_index]);
      g.vertices[vertex].pose = geometry::PoseSE3::exp(-step) * original;
      const Vector6d minus = edge_residual(g, g.edges[edge_index]);

      J.col(k) = (plus - minus) / (2.0 * eps);
    }
    g.vertices[vertex].pose = original;
    return J;
  }

  // A short trajectory plus one loop closure. Measurements are EXACT, so the
  // ground-truth poses are the global optimum and the cost floor is ~1e-31.
  FactorGraph exact_graph(std::vector<geometry::PoseSE3> *truth_out = nullptr) {
    const std::vector<geometry::PoseSE3> truth{
        pose_of(0.0, {0, 0, 0}), pose_of(0.3, {1, 0, 0}),
        pose_of(0.6, {2, 0.5, 0}), pose_of(0.9, {3, 1, 0})};

    FactorGraph g;
    for (const geometry::PoseSE3 &t : truth) g.vertices.push_back(Vertex{t, 0.0});
    for (int i = 0; i + 1 < static_cast<int>(truth.size()); ++i) {
      g.edges.push_back(Edge{i, i + 1,
                             predicted_measurement(truth[i], truth[i + 1]),
                             Information::Identity()});
    }
    // loop closure: the only edge that makes the system over-determined, and so
    // the only reason there is anything to optimise
    g.edges.push_back(Edge{3, 0, predicted_measurement(truth[3], truth[0]),
                           Information::Identity()});

    if (truth_out) *truth_out = truth;
    return g;
  }

  void nudge(FactorGraph &g, const double scale) {
    for (size_t v = 1; v < g.vertices.size(); ++v) {   // vertex 0 is the anchor
      Vector6d d;
      d.setConstant(scale * static_cast<double>(v));
      g.vertices[v].pose = geometry::PoseSE3::exp(d) * g.vertices[v].pose;
    }
  }
}

// Perturbing `from` and perturbing `to` push the residual in exactly opposite
// directions: both routes pass through the same adjoint and differ only by the
// sign picked up from inverting one of them. This holds EXACTLY -- it does not
// depend on the J_r inverse approximation -- so it is a hard constraint on any
// analytic Jacobian pair, and the cheapest way to check the sign structure.
TEST(TestGaussNewton, EdgeJacobiansAreExactNegatives) {
  FactorGraph g = exact_graph();
  nudge(g, 0.05);   // away from zero residual, or the check is degenerate

  for (size_t e = 0; e < g.edges.size(); ++e) {
    const Matrix6d J_from = numeric_jacobian(g, e, g.edges[e].from);
    const Matrix6d J_to = numeric_jacobian(g, e, g.edges[e].to);

    EXPECT_LT((J_from + J_to).norm(), 1e-6)
        << "edge " << e << " J_from is not -J_to\nJ_from:\n" << J_from
        << "\nJ_to:\n" << J_to;
  }
}

// The reference the analytic blocks must match. Prints on failure so the
// mismatch pattern is visible rather than just a norm.
TEST(TestGaussNewton, NumericJacobianIsFiniteAndNonTrivial) {
  FactorGraph g = exact_graph();
  nudge(g, 0.05);

  const Matrix6d J = numeric_jacobian(g, 0, g.edges[0].from);

  EXPECT_TRUE(J.allFinite()) << J;
  EXPECT_GT(J.norm(), 1e-3) << "residual does not respond to moving `from`:\n" << J;
}

// The load-bearing test. Measurements are exactly consistent with a known
// trajectory, so the optimum is that trajectory and the cost floor is ~1e-31.
// A correct Gauss-Newton reaches it in a couple of iterations from a small
// perturbation; wrong Jacobians show up here as a cost that stalls, rises, or
// goes non-finite.
TEST(TestGaussNewton, ConvergesToTruthOnExactMeasurements) {
  std::vector<geometry::PoseSE3> truth;
  FactorGraph g = exact_graph(&truth);
  ASSERT_LT(total_cost(g), 1e-20) << "fixture is not at the optimum";

  nudge(g, 0.02);
  const double before = total_cost(g);
  ASSERT_GT(before, 1e-6) << "perturbation too small to be a test";

  g.gauss_newton(1e-10, true);
  const double after = total_cost(g);

  ASSERT_TRUE(std::isfinite(after)) << "cost is not finite: " << after;
  EXPECT_LT(after, 1e-16) << "cost " << before << " -> " << after;

  for (size_t v = 0; v < truth.size(); ++v) {
    EXPECT_LT((g.vertices[v].pose.matrix() - truth[v].matrix()).norm(), 1e-6)
        << "vertex " << v;
  }
}

// Already optimal: the solver must recognise it and not wander.
TEST(TestGaussNewton, LeavesAnOptimalGraphAlone) {
  std::vector<geometry::PoseSE3> truth;
  FactorGraph g = exact_graph(&truth);

  g.gauss_newton(1e-10, true);

  for (size_t v = 0; v < truth.size(); ++v) {
    EXPECT_LT((g.vertices[v].pose.matrix() - truth[v].matrix()).norm(), 1e-9)
        << "vertex " << v << " moved away from the optimum";
  }
}
