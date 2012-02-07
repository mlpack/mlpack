/**
 * @file lrsdp_test.cpp
 * @author Ryan Curtin
 *
 * Tests for LR-SDP (core/optimizers/lrsdp/).
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lrsdp/lrsdp.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(LRSDPTest);

/**
 * Extremely simple test case for the Lovasz-Theta semidefinite program.
 *
BOOST_AUTO_TEST_CASE(ExtremelySimpleLovaszThetaSDP)
{
  // Manually create the LRSDP object and set its constraints.
  LRSDP lovasz;

  // C = -(e e^T) = -ones().
  lovasz.C().ones(2, 2);
  lovasz.C() *= -1;

  // b_0 = 1; b_1 = 0.
  lovasz.B().zeros(2);
  lovasz.B()[0] = 1;

  // A_0 = I_n.
  lovasz.A().push_back(arma::eye<arma::mat>(2, 2));

  // A_1 = 1 - I_n.
  lovasz.A().push_back(1 - arma::eye<arma::mat>(2, 2));

  // Now generate the initial point.
  arma::mat coordinates(2, 2);

  double r = 0.5 + sqrt(4.25); // 2 constraints.

  coordinates(0, 0) = sqrt(1.0 / r) + sqrt(0.25);
  coordinates(0, 1) = sqrt(0.25);
  coordinates(1, 0) = sqrt(0.25);
  coordinates(1, 1) = sqrt(1.0 / r) + sqrt(0.25);

  // Now that we have an initial point, run the optimization.
  double finalValue = lovasz.Optimize(coordinates);

  arma::mat x = coordinates * trans(coordinates);

  BOOST_REQUIRE_CLOSE(finalValue, -1.0, 1e-5);

  BOOST_REQUIRE_CLOSE(x(0, 0) + x(1, 1), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(x(0, 1), 1e-8);
  BOOST_REQUIRE_SMALL(x(1, 0), 1e-8);
}*/

/**
 * Prepare an LRSDP object to solve the Lovasz-Theta SDP in the manner detailed
 * in Monteiro + Burer 2004.  The list of edges in the graph must be given; that
 * is all that is necessary to set up the problem.  A matrix which will contain
 * initial point coordinates should be given also.
 */
void setupLovaszTheta(const arma::mat& edges,
                      LRSDP& lovasz,
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

  lovasz = LRSDP(coordinates);

  // C = -(e e^T) = -ones().
  lovasz.C().ones(vertices, vertices);
  lovasz.C() *= -1;

  // b_0 = 1; else = 0.
  lovasz.B().zeros(edges.n_cols);
  lovasz.B()[0] = 1;

  // All of the matrices will just contain coordinates because they are
  // super-sparse (two entries each).  Except for A_0, which is I_n.
  lovasz.AModes().ones(edges.n_cols);
  lovasz.AModes()[0] = 0;

  // A_0 = I_n.
  lovasz.A().push_back(arma::eye<arma::mat>(vertices, vertices));

  // A_ij only has ones at (i, j) and (j, i) and 1 elsewhere.
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    arma::mat a(2, 2);

    a(0, 0) = edges(0, i);
    a(1, 0) = edges(1, i);
    a(0, 1) = edges(1, i);
    a(1, 1) = edges(0, i);

    lovasz.A().push_back(a);
  }
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
  LRSDP lovasz;

  setupLovaszTheta(edges, lovasz, coordinates);

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
 * hamming6-4.co test case for Lovasz-Theta LRSDP.
 * See Monteiro and Burer 2004.
 *
BOOST_AUTO_TEST_CASE(Hamming64LovaszThetaSDP)
{
  // Load the edges.
  arma::mat edges;
  data::Load("hamming6-4.csv", edges, true);

  // The LRSDP itself and the initial point.
  arma::mat coordinates;
  LRSDP lovasz;

  setupLovaszTheta(edges, lovasz, coordinates);

  double finalValue = lovasz.Optimize(coordinates);

  // Final value taken from Monteiro + Burer 2004.
  BOOST_REQUIRE_CLOSE(finalValue, -5.333333, 1e-5);

  // Now ensure that all the constraints are satisfied.
  arma::mat rrt = coordinates * trans(coordinates);
  BOOST_REQUIRE_CLOSE(trace(rrt), 1.0, 1e-5);

  // All those edge constraints...
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    BOOST_REQUIRE_SMALL(rrt(edges(0, i), edges(1, i)), 1e-5);
    BOOST_REQUIRE_SMALL(rrt(edges(1, i), edges(0, i)), 1e-5);
  }
}*/

/**
 * keller4.co test case for Lovasz-Theta LRSDP.
 * See Monteiro and Burer 2004.
 *
BOOST_AUTO_TEST_CASE(Keller4LovaszThetaSDP)
{
  // Load the edges.
  arma::mat edges;
  data::Load("keller4.csv", edges, true);

  // The LRSDP itself and the initial point.
  arma::mat coordinates;
  LRSDP lovasz;

  setupLovaszTheta(edges, lovasz, coordinates);

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
