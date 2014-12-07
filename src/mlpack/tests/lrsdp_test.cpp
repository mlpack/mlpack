/**
 * @file lrsdp_test.cpp
 * @author Ryan Curtin
 *
 * Tests for LR-SDP (core/optimizers/lrsdp/).
 *
 * This file is part of MLPACK 1.0.11.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
  lovasz.C().ones(vertices, vertices);
  lovasz.C() *= -1;

  // b_0 = 1; else = 0.
  lovasz.B().zeros(edges.n_cols);
  lovasz.B()[0] = 1;

  // All of the matrices will just contain coordinates because they are
  // super-sparse (two entries each).  Except for A_0, which is I_n.
  lovasz.AModes().ones();
  lovasz.AModes()[0] = 0;

  // A_0 = I_n.
  lovasz.A()[0].eye(vertices, vertices);

  // A_ij only has ones at (i, j) and (j, i) and 1 elsewhere.
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    arma::mat a(3, 2);

    a(0, 0) = edges(0, i);
    a(1, 0) = edges(1, i);
    a(2, 0) = 1;

    a(0, 1) = edges(1, i);
    a(1, 1) = edges(0, i);
    a(2, 1) = 1;

    lovasz.A()[i + 1] = a;
  }

  // Set the Lagrange multipliers right.
  lovasz.AugLag().Lambda().ones(edges.n_cols);
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

  LRSDP lovasz(edges.n_cols + 1, coordinates);

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
