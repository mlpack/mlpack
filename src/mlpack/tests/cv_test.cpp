/**
 * @file cv_test.cpp
 *
 * Unit tests for the cross-validation module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/mse.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack::ann;
using namespace mlpack::cv;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(CVTest);

/*
 * Test the accuracy metric.
 */
BOOST_AUTO_TEST_CASE(AccuracyTest)
{
  // Making linearly separable data.
  arma::mat data =
    arma::mat("1 0; 2 0; 3 0; 4 0; 5 0; 1 1; 2 1; 3 1; 4 1; 5 1").t();
  arma::Row<size_t> trainingLabels("0 0 0 0 0 1 1 1 1 1");

  LogisticRegression<> lr(data, trainingLabels);

  arma::Row<size_t> labels("0 0 1 0 0 1 0 1 0 1"); // 70%-correct labels

  BOOST_REQUIRE_CLOSE(Accuracy::Evaluate(lr, data, labels), 0.7, 1e-5);
}

/*
 * Test the mean squared error.
 */
BOOST_AUTO_TEST_CASE(MSETest)
{
  // Making two points that define the linear function f(x) = x - 1
  arma::mat trainingData("0 1");
  arma::rowvec trainingResponses("-1 0");

  LinearRegression lr(trainingData, trainingResponses);

  // Making three responses that differ from the correct ones by 0, 1, and 2
  // respectively
  arma::mat data("2 3 4");
  arma::rowvec responses("1 3 5");

  double expectedMSE = (0 * 0 + 1 * 1 + 2 * 2) / 3.0;

  BOOST_REQUIRE_CLOSE(MSE::Evaluate(lr, data, responses), expectedMSE, 1e-5);
}

/*
 * Test the mean squared error with matrix responses.
 */
BOOST_AUTO_TEST_CASE(MSEMatResponsesTest)
{
  arma::mat data("1 2");
  arma::mat trainingResponses("1 2; 3 4");

  FFN<MeanSquaredError<>, ZeroInitialization> ffn;
  ffn.Add<Linear<>>(1, 2);
  ffn.Add<IdentityLayer<>>();

  RMSProp<decltype(ffn)> opt(ffn, 0.2);
  opt.Shuffle() = false;
  ffn.Train(data, trainingResponses, opt);

  // Making four responses that differ from the correct ones by 0, 1, 2 and 3
  // respectively
  arma::mat responses("1 3; 5 7");

  double expectedMSE = (0 * 0 + 1 * 1 + 2 * 2 + 3 * 3) / 4.0;

  BOOST_REQUIRE_CLOSE(MSE::Evaluate(ffn, data, responses), expectedMSE, 1e-1);
}

BOOST_AUTO_TEST_SUITE_END();
