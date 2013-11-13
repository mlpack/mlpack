/**
 * @file logistic_regression_test.cpp
 * @author Ryan Curtin
 *
 * Test for LogisticFunction and LogisticRegression.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(LogisticRegressionTest);

/**
 * Test the LogisticFunction on a simple set of points.
 */
BOOST_AUTO_TEST_CASE(LogisticRegressionFunctionEvaluate)
{
  // Very simple fake dataset.
  arma::mat data("1 1 1;" // Fake row for intercept.
                 "1 2 3;"
                 "1 2 3");
  arma::vec responses("1 1 0");

  // Create a LogisticRegressionFunction.
  LogisticRegressionFunction lrf(data, responses, 0.0 /* no regularization */);

  // These were hand-calculated using Octave.
  BOOST_REQUIRE_CLOSE(lrf.Evaluate(arma::vec("1 1 1")), 7.0562141665, 1e-5);
  BOOST_REQUIRE_CLOSE(lrf.Evaluate(arma::vec("0 0 0")), 2.0794415417, 1e-5);
  BOOST_REQUIRE_CLOSE(lrf.Evaluate(arma::vec("-1 -1 -1")), 8.0562141665, 1e-5);
  BOOST_REQUIRE_CLOSE(lrf.Evaluate(arma::vec("200 -40 -40")), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(lrf.Evaluate(arma::vec("200 -80 0")), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(lrf.Evaluate(arma::vec("200 -100 20")), 0.0, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
