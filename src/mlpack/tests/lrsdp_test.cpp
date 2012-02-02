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
 */
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

  const size_t vertices = max(max(edges)) + 1;

  const size_t m = edges.n_cols + 1;
  float r = 0.5 + sqrt(0.25 + 2 * m);
  if (ceil(r) > vertices)
    r = vertices; // An upper bound on the dimension.

  arma::mat coordinates(vertices, ceil(r));

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

  LRSDP lovasz;

  // C = -(e e^T) = -ones().
  lovasz.C().ones(vertices, vertices);
  lovasz.C() *= -1;

  // b_0 = 1; else = 0.
  lovasz.B().zeros(vertices);
  lovasz.B()[0] = 1;

  // A_0 = I_n.
  lovasz.A().push_back(arma::eye<arma::mat>(vertices, vertices));

  // A_ij only has ones at (i, j) and (j, i) and 1 elsewhere.
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    arma::mat a;
    a.zeros(vertices, vertices);

    a(edges(0, i), edges(1, i)) = 1;
    a(edges(1, i), edges(0, i)) = 1;

    lovasz.A().push_back(a);
  }

  double finalValue = lovasz.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(finalValue, -14.0, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
