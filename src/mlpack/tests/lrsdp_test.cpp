/**
 * @file lrsdp_test.cpp
 * @author Ryan Curtin
 *
 * Tests for LR-SDP (core/optimizers/sdp/).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sdp/lrsdp.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(LRSDPTest);

/**
 * Create a Lovasz-Theta initial point.
 */
void CreateLovaszThetaInitialPoint(const arma::mat& edges,
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
void SetupLovaszTheta(const arma::mat& edges,
                      LRSDP<SDP<arma::mat>>& lovasz)
{
  // Get the number of vertices in the problem.
  const size_t vertices = max(max(edges)) + 1;

  // C = -(e e^T) = -ones().
  lovasz.SDP().C().ones(vertices, vertices);
  lovasz.SDP().C() *= -1;

  // b_0 = 1; else = 0.
  lovasz.SDP().SparseB().zeros(edges.n_cols + 1);
  lovasz.SDP().SparseB()[0] = 1;

  // A_0 = I_n.
  lovasz.SDP().SparseA()[0].eye(vertices, vertices);

  // A_ij only has ones at (i, j) and (j, i) and 0 elsewhere.
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    lovasz.SDP().SparseA()[i + 1].zeros(vertices, vertices);
    lovasz.SDP().SparseA()[i + 1](edges(0, i), edges(1, i)) = 1.;
    lovasz.SDP().SparseA()[i + 1](edges(1, i), edges(0, i)) = 1.;
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

  CreateLovaszThetaInitialPoint(edges, coordinates);

  LRSDP<SDP<arma::mat>> lovasz(edges.n_cols + 1, 0, coordinates);

  SetupLovaszTheta(edges, lovasz);

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

/**
 * Create an unweighted graph laplacian from the edges.
 */
void CreateSparseGraphLaplacian(const arma::mat& edges,
                                arma::sp_mat& laplacian)
{
  // Get the number of vertices in the problem.
  const size_t vertices = max(max(edges)) + 1;

  laplacian.zeros(vertices, vertices);

  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    laplacian(edges(0, i), edges(1, i)) = -1.0;
    laplacian(edges(1, i), edges(0, i)) = -1.0;
  }

  for (size_t i = 0; i < vertices; ++i)
  {
    laplacian(i, i) = -arma::accu(laplacian.row(i));
  }
}

BOOST_AUTO_TEST_CASE(ErdosRenyiRandomGraphMaxCutSDP)
{
  // Load the edges.
  arma::mat edges;
  data::Load("erdosrenyi-n100.csv", edges, true);

  arma::sp_mat laplacian;
  CreateSparseGraphLaplacian(edges, laplacian);

  float r = 0.5 + sqrt(0.25 + 2 * edges.n_cols);
  if (ceil(r) > laplacian.n_rows)
    r = laplacian.n_rows;

  // initialize coordinates to a feasible point
  arma::mat coordinates(laplacian.n_rows, ceil(r));
  coordinates.zeros();
  for (size_t i = 0; i < coordinates.n_rows; ++i)
  {
    coordinates(i, i % coordinates.n_cols) = 1.;
  }

  LRSDP<SDP<arma::sp_mat>> maxcut(laplacian.n_rows, 0, coordinates);
  maxcut.SDP().C() = laplacian;
  maxcut.SDP().C() *= -1.; // need to minimize the negative
  maxcut.SDP().SparseB().ones(laplacian.n_rows);
  for (size_t i = 0; i < laplacian.n_rows; ++i)
  {
    maxcut.SDP().SparseA()[i].zeros(laplacian.n_rows, laplacian.n_rows);
    maxcut.SDP().SparseA()[i](i, i) = 1.;
  }

  const double finalValue = maxcut.Optimize(coordinates);
  const arma::mat rrt = coordinates * trans(coordinates);

  for (size_t i = 0; i < laplacian.n_rows; ++i)
  {
    BOOST_REQUIRE_CLOSE(rrt(i, i), 1., 1e-5);
  }

  // Final value taken by solving with Mosek
  BOOST_REQUIRE_CLOSE(finalValue, -3672.7, 1e-1);
}

/*
 * Test a nuclear norm minimization SDP.
 *
 * Specifically, fix an unknown m x n matrix X. Our goal is to recover X from p
 * measurements of X, where the i-th measurement is of the form
 *
 *    b_i = dot(A_i, X)
 *
 * where the A_i's have iid entries from Normal(0, 1/p). We do this by solving
 * the the following semi-definite program
 *
 *    min ||X||_* subj to dot(A_i, X) = b_i, i=1,...,p
 *
 * where ||X||_* denotes the nuclear norm (sum of singular values) of X. The
 * equivalent SDP is
 *
 *    min tr(W1) + tr(W2) : [ W1, X ; X', W2 ] is PSD,
 *                          dot(A_i, X) = b_i, i = 1, ..., p
 *
 * For more details on matrix sensing and nuclear norm minimization, see
 *
 *    Guaranteed Minimum-Rank Solutions of Linear Matrix Equations via Nuclear
 *    Norm Minimization.
 *    Benjamin Recht, Maryam Fazel, Pablo Parrilo.
 *    SIAM Review 2010.
 *
 */
BOOST_AUTO_TEST_CASE(GaussianMatrixSensingSDP)
{
  arma::mat Xorig, A;

  // read the unknown matrix X and the measurement matrices A_i in
  data::Load("sensing_X.csv", Xorig, true, false);
  data::Load("sensing_A.csv", A, true, false);

  const size_t m = Xorig.n_rows;
  const size_t n = Xorig.n_cols;
  const size_t p = A.n_rows;
  assert(A.n_cols == m * m);

  arma::vec b(p);
  for (size_t i = 0; i < p; ++i)
  {
    const arma::mat Ai = arma::reshape(A.row(i), n, m);
    b(i) = arma::dot(trans(Ai), Xorig);
  }

  float r = 0.5 + sqrt(0.25 + 2 * p);
  if (ceil(r) > m + n)
    r = m + n;

  arma::mat coordinates;
  coordinates.eye(m + n, ceil(r));

  LRSDP<SDP<arma::sp_mat>> sensing(0, p, coordinates);
  sensing.SDP().C().eye(m + n, m + n);
  sensing.SDP().DenseB() = 2. * b;

  const auto block_rows = arma::span(0, m - 1);
  const auto block_cols = arma::span(m, m + n - 1);

  for (size_t i = 0; i < p; ++i)
  {
    const arma::mat Ai = arma::reshape(A.row(i), n, m);
    sensing.SDP().DenseA()[i].zeros(m + n, m + n);
    sensing.SDP().DenseA()[i](block_rows, block_cols) = trans(Ai);
    sensing.SDP().DenseA()[i](block_cols, block_rows) = Ai;
  }

  double finalValue = sensing.Optimize(coordinates);
  BOOST_REQUIRE_CLOSE(finalValue, 44.7550132629, 1e-1);

  const arma::mat rrt = coordinates * trans(coordinates);
  for (size_t i = 0; i < p; ++i)
  {
    const arma::mat Ai = arma::reshape(A.row(i), n, m);
    const double measurement =
        arma::dot(trans(Ai), rrt(block_rows, block_cols));
    BOOST_REQUIRE_CLOSE(measurement, b(i), 1e-3);
  }

  // check matrix recovery
  const double err = arma::norm(Xorig - rrt(block_rows, block_cols), "fro") /
      arma::norm(Xorig, "fro");
  BOOST_REQUIRE_SMALL(err, 1e-3);
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

  CreateLovaszThetaInitialPoint(edges, coordinates);

  LRSDP<SDP<arma::mat>> lovasz(edges.n_cols, coordinates);

  SetupLovaszTheta(edges, lovasz);

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
