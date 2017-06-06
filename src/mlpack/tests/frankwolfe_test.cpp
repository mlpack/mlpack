/**
 * @file frankwolfe_test.cpp
 * @author Chenzhe Diao
 *
 * Test file for Frank-Wolfe optimizer.
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
#include <mlpack/core/optimizers/fw/test_func_sq.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
//using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(FrankWolfeTest);

BOOST_AUTO_TEST_CASE(OMPTest)
{
  mat A("1, 0, 0, 0.2, -0.15; 0, 1, 0, -0.03, -0.3; 0, 0, 1, 0.1, 0.1");
  vec b("1; 1; 0");

  TestFuncSq f(A, b);
  ConstrLpBallSolver linear_constr_solver(1);
  UpdateSpan<TestFuncSq> update_rule(f);

  OMP s(f, linear_constr_solver, update_rule);

  arma::vec coordinates("0; 0; 0; 0; 0");
  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[0]-1, 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[1]-1, 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[3], 1e-10);
  BOOST_REQUIRE_SMALL(coordinates[4], 1e-10);
}


BOOST_AUTO_TEST_SUITE_END();
