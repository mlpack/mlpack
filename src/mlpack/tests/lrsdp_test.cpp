/**
 * @file lrsdp_test.cpp
 * @author Ryan Curtin
 *
 * Tests for LR-SDP (core/optimizers/lrsdp/).
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lrsdp/lrsdp.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(LRSDPTest);

/**
 * Create a Lovasz-Theta initial point.
 */
void createLovaszThetaInitialPoint(const arma::mat& edges,
                                   arma::mat& coordinates)
{
  // Get the number of vertices in the problem.
  const size_t vertices = max(max(edges)) + 1;

  const size_t m = edges.n_cols + 1;
  float r = 0.5 + sqrt(0.25 + 2 * m);
  if (ceil(r) > vertices)
    r = vertices; // An upper bound on the dimension.

  coordinates.set_size(vertices, ceil(r));

  // Now we set the entries of the initial matrix according to the formula given
  // in Section 4 of Monteiro and Burer.
  for (size_t i = 0; i < vertices; ++i)
  {
    for (size_t j = 0; j < ceil(r); ++j)
    {
      if (i == j)
        coordinates(i, j) = sqrt(1.0 / r) + sqrt(1.0 / (vertices * m));
      else
        coordinates(i, j) = sqrt(1.0 / (vertices * m));
    }
  }
}

/**
 * Prepare an LRSDP object to solve the Lovasz-Theta SDP in the manner detailed
 * in Monteiro + Burer 2004.  The list of edges in the graph must be given; that
 * is all that is necessary to set up the problem.  A matrix which will contain
 * initial point coordinates should be given also.
 */
void setupLovaszTheta(const arma::mat& edges,
                      LRSDP& lovasz)
{
  // Get the number of vertices in the problem.
  const size_t vertices = max(max(edges)) + 1;

  // C = -(e e^T) = -ones().
  lovasz.DenseC().ones(vertices, vertices);
  lovasz.DenseC() *= -1;

  // b_0 = 1; else = 0.
  lovasz.SparseB().zeros(edges.n_cols + 1);
  lovasz.SparseB()[0] = 1;

  // A_0 = I_n.
  lovasz.SparseA()[0].eye(vertices, vertices);

  // A_ij only has ones at (i, j) and (j, i) and 0 elsewhere.
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    lovasz.SparseA()[i + 1].zeros(vertices, vertices);
    lovasz.SparseA()[i + 1](edges(0, i), edges(1, i)) = 1.;
    lovasz.SparseA()[i + 1](edges(1, i), edges(0, i)) = 1.;
  }

  // Set the Lagrange multipliers right.
  lovasz.AugLag().Lambda().ones(edges.n_cols + 1);
  lovasz.AugLag().Lambda() *= -1;
  lovasz.AugLag().Lambda()[0] = -double(vertices);
}

/**
 * johnson8-4-4.co test case for Lovasz-Theta LRSDP.
 * See Monteiro and Burer 2004.
 */
BOOST_AUTO_TEST_CASE(Johnson844LovaszThetaSDP)
{
  // Load the edges.
  arma::mat edges;
  data::Load("johnson8-4-4.csv", edges, true);

  // The LRSDP itself and the initial point.
  arma::mat coordinates;

  createLovaszThetaInitialPoint(edges, coordinates);

  LRSDP lovasz(edges.n_cols + 1, 0, coordinates);

  setupLovaszTheta(edges, lovasz);

  double finalValue = lovasz.Optimize(coordinates);

  // Final value taken from Monteiro + Burer 2004.
  BOOST_REQUIRE_CLOSE(finalValue, -14.0, 1e-5);

  // Now ensure that all the constraints are satisfied.
  arma::mat rrt = coordinates * trans(coordinates);
  BOOST_REQUIRE_CLOSE(trace(rrt), 1.0, 1e-5);

  // All those edge constraints...
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    BOOST_REQUIRE_SMALL(rrt(edges(0, i), edges(1, i)), 1e-5);
    BOOST_REQUIRE_SMALL(rrt(edges(1, i), edges(0, i)), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(GaussianMatrixSensingSDP)
{
  arma::mat Xorig, A, bmat;

  data::Load("sensing_X.csv", Xorig, true, false);
  data::Load("sensing_A.csv", A, true, false);
  data::Load("sensing_b.csv", bmat, true, false);

  arma::vec b = bmat;

  const size_t m = Xorig.n_rows;
  const size_t n = Xorig.n_cols;
  const size_t p = b.n_elem;

  assert(A.n_rows == p);
  assert(A.n_cols == m * m);

  float r = 0.5 + sqrt(0.25 + 2 * p);
  if (ceil(r) > m + n)
    r = m + n;

  std::srand(0x0FAE8543);
  arma::mat coordinates = arma::randu<arma::mat>(m + n, ceil(r));
  std::srand(std::time(nullptr));

  LRSDP sensing(0, p, coordinates);
  sensing.SparseC().eye(m + n, m + n);
  sensing.DenseB() = 2. * b;
  for (size_t i = 0; i < p; ++i)
  {
    const auto rows = arma::span(0, m - 1);
    const auto cols = arma::span(m, m + n - 1);
    const auto Ai = arma::reshape(A.row(i), n, m);
    sensing.DenseA()[i].zeros(m + n, m + n);
    sensing.DenseA()[i](rows, cols) = trans(Ai);
    sensing.DenseA()[i](cols, rows) = Ai;
  }

  double finalValue = sensing.Optimize(coordinates);
  BOOST_REQUIRE_CLOSE(finalValue, 44.7550132629, 1e-1);

  const arma::mat rrt = coordinates * trans(coordinates);
  for (size_t i = 0; i < p; ++i)
  {
    const auto rows = arma::span(0, m - 1);
    const auto cols = arma::span(m, m + n - 1);
    const auto Ai = arma::reshape(A.row(i), n, m);
    BOOST_REQUIRE_CLOSE(arma::dot(trans(Ai), rrt(rows, cols)), b(i), 1e-3);
  }
}

/**
 * keller4.co test case for Lovasz-Theta LRSDP.
 * This is commented out because it takes a long time to run.
 * See Monteiro and Burer 2004.
 *
BOOST_AUTO_TEST_CASE(Keller4LovaszThetaSDP)
{
  // Load the edges.
  arma::mat edges;
  data::Load("keller4.csv", edges, true);

  // The LRSDP itself and the initial point.
  arma::mat coordinates;

  createLovaszThetaInitialPoint(edges, coordinates);

  LRSDP lovasz(edges.n_cols, coordinates);

  setupLovaszTheta(edges, lovasz);

  double finalValue = lovasz.Optimize(coordinates);

  // Final value taken from Monteiro + Burer 2004.
  BOOST_REQUIRE_CLOSE(finalValue, -14.013, 1e-2); // Not as much precision...
  // The SB method came to -14.013, but M&B's method only came to -14.005.

  // Now ensure that all the constraints are satisfied.
  arma::mat rrt = coordinates * trans(coordinates);
  BOOST_REQUIRE_CLOSE(trace(rrt), 1.0, 1e-5);

  // All those edge constraints...
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    BOOST_REQUIRE_SMALL(rrt(edges(0, i), edges(1, i)), 1e-3);
    BOOST_REQUIRE_SMALL(rrt(edges(1, i), edges(0, i)), 1e-3);
  }
}*/

BOOST_AUTO_TEST_SUITE_END();
