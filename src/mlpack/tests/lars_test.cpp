/**
 * @file lars_test.cpp
 *
 * Test for LARS
 */

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.


#include <armadillo>
#include <mlpack/methods/lars/lars.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(LARSTest);

void GenerateProblem(arma::mat& X, arma::vec& y, size_t nPoints, size_t nDims)
{
  X = arma::randn(nPoints, nDims);
  arma::vec beta = arma::randn(nDims, 1);
  y = X * beta;
}


void VerifyCorrectness(arma::vec beta, arma::vec errCorr, double lambda)
{
  size_t nDims = beta.n_elem;
  const double tol = 1e-12;
  for(size_t j = 0; j < nDims; j++)
  {
    if (beta(j) == 0)
    {
      // make sure that |errCorr(j)| <= lambda
      BOOST_REQUIRE_SMALL(std::max(fabs(errCorr(j)) - lambda, 0.0), tol);
    }
    else if (beta(j) < 0)
    {
      // make sure that errCorr(j) == lambda
      BOOST_REQUIRE_SMALL(errCorr(j) - lambda, tol);
    }
    else
    { // beta(j) > 0
      // make sure that errCorr(j) == -lambda
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
    arma::vec sortedAbsCorr = sort(abs(trans(X) * y));
    double lambda1 = sortedAbsCorr(nDims/2);
    double lambda2;
    if (elasticNet)
      lambda2 = lambda1 / 2;
    else
      lambda2 = 0;


    LARS lars(useCholesky, lambda1, lambda2);
    lars.DoLARS(X, y);

    arma::vec betaOpt;
    lars.Solution(betaOpt);
    arma::vec errCorr = (arma::trans(X) * X + lambda2 *
        arma::eye(nDims, nDims)) * betaOpt - arma::trans(X) * y;

    VerifyCorrectness(betaOpt, errCorr, lambda1);
  }
}


BOOST_AUTO_TEST_CASE(LARSTestLassoCholesky)
{
  LassoTest(100, 10, true, false);
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

BOOST_AUTO_TEST_SUITE_END();
