/**
 * @file scd_test.cpp
 * @author Shikhar Bhardwaj
 *
 * Test file for SCD (stochastic coordinate descent).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/scd/scd.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(SCDTest);

BOOST_AUTO_TEST_CASE(PreCalcSCDTest)
{
  arma::mat predictors("0 0 0.4; 0 0 0.6; 0 0.3 0; 0.2 0 0; 0.2 -0.5 0;");
  arma::Row<size_t> responses("1  1  0;");


  LogisticRegressionFunction<arma::mat> f(predictors, responses, 0.0001);

  SCD<> s(0.01, 20000, 1e-5);
  arma::mat iterate = f.InitialPoint();

  double objective = s.Optimize(f, iterate);

  BOOST_REQUIRE_LE(objective, 0.055);
}

BOOST_AUTO_TEST_SUITE_END();
