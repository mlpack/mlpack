/**
 * @file proximal_test.cpp
 * @author Chenzhe Diao
 *
 * Test file for proximal optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/proximal/proximal.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(ProximalTest);

/**
 * Approximate vector using a vector with l1 norm small than or equal to tau.
 */
BOOST_AUTO_TEST_CASE(ProjectToL1)
{
  int D = 100;  // Dimension of the problem.

  // Norm of L1 ball.
  double tau1 = 1.5;
  double tau2 = 0.5;

  // Vector to be projected, with unit l1 norm.
  vec v = randu<vec>(D);
  v = normalise(v, 1);

  // v is inside the l1 ball, so the projection will not change v.
  vec v1 = v;
  Proximal::ProjectToL1Ball(v1, tau1);
  BOOST_REQUIRE_SMALL(norm(v - v1, 2), 1e-10);

  // v is outside the l1 ball, so the projection should find the closest.
  vec v2 = v;
  Proximal::ProjectToL1Ball(v2, tau2);
  double distance = norm(v2 - v, 2);
  for (size_t i = 1; i < 1000; i++)
  {
    // Randomly generate a vector on the surface of the l1 ball with norm tau2.
    vec vSurface = randu<vec>(D);
    vSurface = tau2 * normalise(vSurface, 1);

    double distanceNew = norm(vSurface - v, 2);

    BOOST_REQUIRE_GE(distanceNew, distance);
  }
}

/**
 * Approximate a vector with a tau-sparse vector.
 */
BOOST_AUTO_TEST_CASE(ProjectToL0)
{
  int D = 100;  // Dimension of the problem.
  int tau = 25; // Sparsity requirement.

  // Vector to be projected.
  vec v = randn<vec>(D);

  vec v0 = v;
  Proximal::ProjectToL0Ball(v0, tau);
  double distance = norm(v0 - v, 2);

  for (size_t i = 1; i < 1000; i++)
  {
    // Randomly find a subset of the support of v, generate a tau-sparse
    // vector by restricting v to this support.
    uvec indices = linspace<uvec>(0, D - 1, D);
    indices = shuffle(indices);
    indices = indices.head(tau);
    vec vNew = zeros<vec>(D);
    vNew.elem(indices) = v.elem(indices);

    double distanceNew = norm(v - vNew, 2);
    BOOST_REQUIRE_GE(distanceNew, distance);
  }
}

BOOST_AUTO_TEST_SUITE_END();
