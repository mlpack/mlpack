/**
 * @file lars_test.cpp
 *
 * Test for LARS
 * This file is part of MLPACK 1.0.2.
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

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.


#include <armadillo>
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

BOOST_AUTO_TEST_SUITE_END();
