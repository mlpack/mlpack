/**
 * @file function_test.cpp
 * @author Ryan Curtin
 *
 * Test the Function<> class to see that it properly adds functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_function.hpp>
#include <mlpack/core/optimizers/sdp/sdp.hpp>
#include <mlpack/core/optimizers/sdp/lrsdp.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::aux; // For some SFINAE checks.
using namespace mlpack::regression;

/**
 * Utility class with no functions.
 */
class EmptyTestFunction { };

/**
 * Utility class with Evaluate() but no Evaluate().
 */
class EvaluateTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }
};

/**
 * Utility class with Gradient() but no Evaluate().
 */
class GradientTestFunction
{
 public:
  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with Gradient() and Evaluate().
 */
class EvaluateGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with EvaluateWithGradient().
 */
class EvaluateWithGradientTestFunction
{
 public:
  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }
};

/**
 * Utility class with all three functions.
 */
class EvaluateAndWithGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }
};

BOOST_AUTO_TEST_SUITE(FunctionTest);

/**
 * Make sure that an empty class doesn't have any methods added to it.
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientEmptyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<EmptyTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EmptyTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EmptyTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, false);
  BOOST_REQUIRE_EQUAL(hasGradient, false);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we don't add any functions if we only have Evaluate().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientEvaluateOnlyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<EvaluateTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EvaluateTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, false);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we don't add any functions if we only have Gradient().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientGradientOnlyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<GradientTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<GradientTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<GradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, false);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we add EvaluateWithGradient() when we have both Evaluate() and
 * Gradient().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientBothTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we add Evaluate() and Gradient() when we have only
 * EvaluateWithGradient().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientEvaluateWithGradientTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateWithGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateWithGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateWithGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we add no methods when we already have all three.
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientAllThreeTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndWithGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndWithGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndWithGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

BOOST_AUTO_TEST_CASE(LogisticRegressionEvaluateWithGradientTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<LogisticRegressionFunction<>>,
                           EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<LogisticRegressionFunction<>>,
                           GradientConstForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<LogisticRegressionFunction<>>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

BOOST_AUTO_TEST_CASE(SDPTest)
{
  typedef AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>> FunctionType;

  const bool hasEvaluate =
      HasEvaluate<Function<FunctionType>, EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<FunctionType>, GradientConstForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<FunctionType>,
                              EvaluateWithGradientConstForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

BOOST_AUTO_TEST_SUITE_END();
