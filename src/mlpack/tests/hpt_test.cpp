/**
 * @file hpt_test.cpp
 *
 * Tests for the hyper-parameter tuning module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/cv/metrics/mse.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/hpt/bind.hpp>
#include <mlpack/core/hpt/cv_function.hpp>
#include <mlpack/methods/lars/lars.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack::cv;
using namespace mlpack::hpt;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(HPTTest);

/**
 * Test CVFunction runs cross-validation in according with specified bound
 * arguments and passed parameters.
 */
BOOST_AUTO_TEST_CASE(CVFunctionTest)
{
  arma::mat xs = arma::randn(5, 100);
  arma::vec beta = arma::randn(5, 1);
  arma::mat ys = beta.t() * xs + 0.1 * arma::randn(5, 1);

  SimpleCV<LARS, MSE> cv(0.2, xs, ys);

  bool transposeData = true;
  bool useCholesky = false;
  double lambda1 = 1.0;
  double lambda2 = 2.0;

  BoundArg<bool, 1> boundUseCholesky{useCholesky};
  BoundArg<double, 3> boundLambda1{lambda2};
  CVFunction<decltype(cv), 4, BoundArg<bool, 1>, BoundArg<double, 3>>
      cvFun(cv, boundUseCholesky, boundLambda1);

  double expected = cv.Evaluate(transposeData, useCholesky, lambda1, lambda2);
  double actual = cvFun.Evaluate(arma::vec{double(transposeData), lambda1});

  BOOST_REQUIRE_CLOSE(expected, actual, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
