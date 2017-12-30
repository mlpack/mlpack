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
#include <mlpack/core/optimizers/fw/update_full_correction.hpp>
#include <mlpack/core/optimizers/fw/update_classic.hpp>
#include <mlpack/core/optimizers/fw/update_linesearch.hpp>
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
  vec b;
  b << 1 << 1 << 0; // Vector to be sparsely approximated.

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateSpan updateRule;

  OMP s(linearConstrSolver, updateRule);

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
 * Simple test of Orthogonal Matching Pursuit with regularization.
 */
BOOST_AUTO_TEST_CASE(regularizedOMP)
{
  int k = 10;
  mat B1 = 0.1 * eye(k, k);
  mat B2 = 100 * randn(k, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b(k, arma::fill::zeros); // Vector to be sparsely approximated.
  b(0) = 1;
  b(1) = 1;
  vec lambda(A.n_cols);
  for (size_t ii = 0; ii < A.n_cols; ii++)
    lambda(ii) = norm(A.col(ii), 2);

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1, lambda);
  UpdateSpan updateRule;

  OMP s(linearConstrSolver, updateRule);

  vec coordinates = zeros<vec>(2 * k);
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-10);
}

/**
 * Simple test of Orthogonal Matching Pursuit with support prune.
 */
BOOST_AUTO_TEST_CASE(PruneSupportOMP)
{
  // The dictionary is input as columns of A.
  int k = 3;
  mat B1;
  B1 << 1 << 0 << 1 << endr
     << 0 << 1 << 1 << endr
     << 0 << 0 << 1 << endr;
  mat B2 = randu(k, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b;
  b << 1 << 1 << 0; // Vector to be sparsely approximated.

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateSpan updateRule(true);

  OMP s(linearConstrSolver, updateRule);

  vec coordinates = zeros<vec>(k + 3);
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-10);
}

/**
 * Simple test of sparse soluton in atom domain with atom norm constraint.
 */
BOOST_AUTO_TEST_CASE(AtomNormConstraint)
{
  int k = 5;
  mat B1 = eye(3, 3);
  mat B2 = 0.1 * randn(3, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b;
  b << 1 << 1 << 0; // Vector to be sparsely approximated.

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateFullCorrection updateRule(2, 0.2);

  FrankWolfe<ConstrLpBallSolver, UpdateFullCorrection>
    s(linearConstrSolver, updateRule);

  vec coordinates = zeros<vec>(k + 3);
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-10);
}


/**
 * A very simple test of classic Frank-Wolfe algorithm.
 * The constrained domain used is unit lp ball.
 */
BOOST_AUTO_TEST_CASE(ClassicFW)
{
  TestFuncFW f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateClassic updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateClassic>
      s(linearConstrSolver, updateRule);

  vec coordinates = randu<vec>(3);
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[0] - 0.1, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[1] - 0.2, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[2] - 0.3, 1e-4);
}

/**
 * Exactly the same problem with ClassicFW.
 * The update step performs a line search now.
 * It converges much faster.
 */
BOOST_AUTO_TEST_CASE(FWLineSearch)
{
  TestFuncFW f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateLineSearch updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateLineSearch>
      s(linearConstrSolver, updateRule);

  vec coordinates = randu<vec>(3);
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[0] - 0.1, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[1] - 0.2, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[2] - 0.3, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END();
