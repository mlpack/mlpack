/**
 * @file lars_test.cpp
 * @author Nishant Mehta
 *
 * Test for LARS.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.
#include <mlpack/methods/lars/lars.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

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
  const double tol = 1e-10;
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

  for (size_t i = 0; i < 100; i++)
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
    lars.Train(X, y, betaOpt);

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
    lars.Train(X, y, betaOpt);

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
    lars.Train(X, y, betaOpt);

    arma::vec errCorr = (X * X.t()) * betaOpt - X * y;

    // #373: this test fails on i386 only sometimes.
//    LARSVerifyCorrectness(betaOpt, errCorr, lambda1);
  }
}

// Make sure that Predict() provides reasonable enough solutions.
BOOST_AUTO_TEST_CASE(PredictTest)
{
  for (size_t i = 0; i < 2; ++i)
  {
    // Run with both true and false.
    bool useCholesky = bool(i);

    arma::mat X;
    arma::vec y;

    GenerateProblem(X, y, 1000, 100);

    for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.2)
    {
      for (double lambda2 = 0.0; lambda2 < 1.0; lambda2 += 0.2)
      {
        LARS lars(useCholesky, lambda1, lambda2);
        arma::vec betaOpt;
        lars.Train(X, y, betaOpt);

        // Calculate what the actual error should be with these regression
        // parameters.
        arma::vec betaOptPred = (X * X.t()) * betaOpt;
        arma::vec predictions;
        lars.Predict(X, predictions);
        arma::vec adjPred = X * predictions;

        BOOST_REQUIRE_EQUAL(predictions.n_elem, 1000);
        for (size_t i = 0; i < betaOptPred.n_elem; ++i)
        {
          if (std::abs(betaOptPred[i]) < 1e-5)
            BOOST_REQUIRE_SMALL(adjPred[i], 1e-5);
          else
            BOOST_REQUIRE_CLOSE(adjPred[i], betaOptPred[i], 1e-5);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(PredictRowMajorTest)
{
  arma::mat X;
  arma::vec y;
  GenerateProblem(X, y, 1000, 100);

  // Set lambdas to 0.

  LARS lars(false, 0, 0);
  arma::vec betaOpt;
  lars.Train(X, y, betaOpt);

  // Get both row-major and column-major predictions.  Make sure they are the
  // same.
  arma::vec rowMajorPred, colMajorPred;

  lars.Predict(X, colMajorPred);
  lars.Predict(X.t(), rowMajorPred, true);

  BOOST_REQUIRE_EQUAL(colMajorPred.n_elem, rowMajorPred.n_elem);
  for (size_t i = 0; i < colMajorPred.n_elem; ++i)
  {
    if (std::abs(colMajorPred[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(rowMajorPred[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(colMajorPred[i], rowMajorPred[i], 1e-5);
  }
}

/**
 * Make sure that if we train twice, there is no issue.
 */
BOOST_AUTO_TEST_CASE(RetrainTest)
{
  arma::mat origX;
  arma::vec origY;
  GenerateProblem(origX, origY, 1000, 50);

  arma::mat newX;
  arma::vec newY;
  GenerateProblem(newX, newY, 750, 75);

  LARS lars(false, 0.1, 0.1);
  arma::vec betaOpt;
  lars.Train(origX, origY, betaOpt);

  // Now train on new data.
  lars.Train(newX, newY, betaOpt);

  arma::vec errCorr = (newX * trans(newX) + 0.1 *
        arma::eye(75, 75)) * betaOpt - newX * newY;

  LARSVerifyCorrectness(betaOpt, errCorr, 0.1);
}

/**
 * Make sure if we train twice using the Cholesky decomposition, there is no
 * issue.
 */
BOOST_AUTO_TEST_CASE(RetrainCholeskyTest)
{
  arma::mat origX;
  arma::vec origY;
  GenerateProblem(origX, origY, 1000, 50);

  arma::mat newX;
  arma::vec newY;
  GenerateProblem(newX, newY, 750, 75);

  LARS lars(true, 0.1, 0.1);
  arma::vec betaOpt;
  lars.Train(origX, origY, betaOpt);

  // Now train on new data.
  lars.Train(newX, newY, betaOpt);

  arma::vec errCorr = (newX * trans(newX) + 0.1 *
        arma::eye(75, 75)) * betaOpt - newX * newY;

  LARSVerifyCorrectness(betaOpt, errCorr, 0.1);
}

BOOST_AUTO_TEST_SUITE_END();
