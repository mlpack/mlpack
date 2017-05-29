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
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack::cv;
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

BOOST_AUTO_TEST_SUITE_END();
