/**
 * @file lars_test.cpp
 * @author Nishant Mehta
 *
 * Test for LARS.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.
#include <mlpack/methods/lars/lars.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(LARSTest);

void GenerateProblem(arma::mat& X, arma::vec& y, size_t nPoints, size_t nDims)
{
  X = arma::randn(nDims, nPoints);
  arma::vec beta = arma::randn(nDims, 1);
  y = trans(X) * beta;
}


void LARSVerifyCorrectness(arma::vec beta, arma::vec errCorr, double lambda)
{
  size_t nDims = beta.n_elem;
  const double tol = 1e-12;
  for(size_t j = 0; j < nDims; j++)
  {
    if (beta(j) == 0)
    {
      // Make sure that |errCorr(j)| <= lambda.
      BOOST_REQUIRE_SMALL(std::max(fabs(errCorr(j)) - lambda, 0.0), tol);
    }
    else if (beta(j) < 0)
    {
      // Make sure that errCorr(j) == lambda.
      BOOST_REQUIRE_SMALL(errCorr(j) - lambda, tol);
    }
    else // beta(j) > 0
    {
      // Make sure that errCorr(j) == -lambda.
      BOOST_REQUIRE_SMALL(errCorr(j) + lambda, tol);
    }
  }
}


void LassoTest(size_t nPoints, size_t nDims, bool elasticNet, bool useCholesky)
{
  arma::mat X;
  arma::vec y;

  for(size_t i = 0; i < 100; i++)
  {
    GenerateProblem(X, y, nPoints, nDims);

    // Armadillo's median is broken, so...
    arma::vec sortedAbsCorr = sort(abs(X * y));
    double lambda1 = sortedAbsCorr(nDims / 2);
    double lambda2;
    if (elasticNet)
      lambda2 = lambda1 / 2;
    else
      lambda2 = 0;


    LARS lars(useCholesky, lambda1, lambda2);
    arma::vec betaOpt;
    lars.Regress(X, y, betaOpt);

    arma::vec errCorr = (X * trans(X) + lambda2 *
        arma::eye(nDims, nDims)) * betaOpt - X * y;

    LARSVerifyCorrectness(betaOpt, errCorr, lambda1);
  }
}


BOOST_AUTO_TEST_CASE(LARSTestLassoCholesky)
{
  LassoTest(100, 10, false, true);
}


BOOST_AUTO_TEST_CASE(LARSTestLassoGram)
{
  LassoTest(100, 10, false, false);
}


BOOST_AUTO_TEST_CASE(LARSTestElasticNetCholesky)
{
  LassoTest(100, 10, true, true);
}


BOOST_AUTO_TEST_CASE(LARSTestElasticNetGram)
{
  LassoTest(100, 10, true, false);
}

// Ensure that LARS doesn't crash when the data has linearly dependent features
// (meaning that there is a singularity).  This test uses the Cholesky
// factorization.
BOOST_AUTO_TEST_CASE(CholeskySingularityTest)
{
  arma::mat X;
  arma::mat Y;

  data::Load("lars_dependent_x.csv", X);
  data::Load("lars_dependent_y.csv", Y);

  arma::vec y = Y.row(0).t();

  // Test for a couple values of lambda1.
  for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.1)
  {
    LARS lars(true, lambda1, 0.0);
    arma::vec betaOpt;
    lars.Regress(X, y, betaOpt);

    arma::vec errCorr = (X * X.t()) * betaOpt - X * y;

    LARSVerifyCorrectness(betaOpt, errCorr, lambda1);
  }
}

// Same as the above test but with no cholesky factorization.
BOOST_AUTO_TEST_CASE(NoCholeskySingularityTest)
{
  arma::mat X;
  arma::mat Y;

  data::Load("lars_dependent_x.csv", X);
  data::Load("lars_dependent_y.csv", Y);

  arma::vec y = Y.row(0).t();

  // Test for a couple values of lambda1.
  for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.1)
  {
    LARS lars(false, lambda1, 0.0);
    arma::vec betaOpt;
    lars.Regress(X, y, betaOpt);

    arma::vec errCorr = (X * X.t()) * betaOpt - X * y;

    // #373: this test fails on i386 only sometimes.
//    LARSVerifyCorrectness(betaOpt, errCorr, lambda1);
  }
}

BOOST_AUTO_TEST_SUITE_END();
