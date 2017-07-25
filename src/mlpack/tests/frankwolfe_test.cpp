/**
 * @file frankwolfe_test.cpp
 * @author Chenzhe Diao
 *
 * Test file for Frank-Wolfe type optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/fw/frank_wolfe.hpp>
#include <mlpack/core/optimizers/fw/constr_lpball.hpp>
#include <mlpack/core/optimizers/fw/update_span.hpp>
#include <mlpack/core/optimizers/fw/update_classic.hpp>
#include <mlpack/core/optimizers/fw/func_sq.hpp>
#include <mlpack/core/optimizers/fw/test_func_fw.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(FrankWolfeTest);

/**
 * Simple test of Orthogonal Matching Pursuit algorithm.
 */
BOOST_AUTO_TEST_CASE(OMPTest)
{
  int k = 5;
  mat B1 = eye(3, 3);
  mat B2 = 0.1 * randn(3, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b("1; 1; 0"); // Vector to be sparsely approximated.

  FuncSq f(A, b);
  ConstrLpBallSolver linear_constr_solver(1);
  UpdateSpan update_rule;

  OMP s(linear_constr_solver, update_rule);

  vec coordinates = zeros<vec>(k + 3);
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[0] - 1, 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[1] - 1, 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-10);
  for (int ii = 0; ii < k; ++ii)
  {
    BOOST_REQUIRE_SMALL(coordinates[ii + 3], 1e-10);
  }
}

/**
 * A very simple test of classic Frank-Wolfe algorithm.
 * The constrained domain used is unit lp ball.
 */
BOOST_AUTO_TEST_CASE(ClassicFW)
{
  TestFuncFW f;
  double p = 1;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linear_constr_solver(p);
  UpdateClassic update_rule;

  FrankWolfe<ConstrLpBallSolver, UpdateClassic>
      s(linear_constr_solver, update_rule);

  vec coordinates = zeros<vec>(3);
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[0] - 0.1, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[1] - 0.2, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[2] - 0.3, 1e-4);
}


BOOST_AUTO_TEST_SUITE_END();
