/**
 * @file linear_regression_test.cpp
 * @author Eugene Freyman
 *
 * Test mlpackMain() of linear_regression_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "LinearRegression";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/linear_regression/linear_regression_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LRTestFixture
{
 public:
  LRTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LRTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

void ResetSettings()
{
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(LinearRegressionMainTest, LRTestFixture);

/**
 * Training a model with different regularization parameter and ensuring that
 * predictions are different.
 */
BOOST_AUTO_TEST_CASE(LRDifferentLambdas)
{
  // A required minimal difference between solutions.
  const double delta = 0.1;

  arma::mat trainX({1.0, 2.0, 3.0});
  arma::mat testX({4.0});
  arma::rowvec trainY({1.0, 4.0, 9.0});

  SetInputParam("training", trainX);
  SetInputParam("training_responses", trainY);
  SetInputParam("test", testX);
  SetInputParam("lambda", 0.1);

  // The first solution.
  mlpackMain();
  const double testY1 = CLI::GetParam<arma::rowvec>("output_predictions")(0);

  bindings::tests::CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", std::move(testX));
  SetInputParam("lambda", 1.0);

  // The second solution.
  mlpackMain();
  const double testY2 = CLI::GetParam<arma::rowvec>("output_predictions")(0);

  // Second solution has stronger regularization,
  // so the predicted value should be smaller.
  BOOST_REQUIRE_GT(testY1 - delta, testY2);
}


/**
 * Checking two options of specifying responses (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
BOOST_AUTO_TEST_CASE(LRResponsesRepresentation)
{
  constexpr double delta = 1e-5;

  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}});
  arma::mat testX({4.0});
  SetInputParam("training", trainX1);
  SetInputParam("test", testX);

  // The first solution.
  mlpackMain();
  const double testY1 = CLI::GetParam<arma::rowvec>("output_predictions")(0);

  bindings::tests::CleanMemory();
  ResetSettings();

  arma::mat trainX2({1.0, 2.0, 3.0});
  arma::rowvec trainY2({1.0, 4.0, 9.0});
  SetInputParam("training", std::move(trainX2));
  SetInputParam("training_responses", std::move(trainY2));
  SetInputParam("test", std::move(testX));

  // The second solution.
  mlpackMain();
  const double testY2 = CLI::GetParam<arma::rowvec>("output_predictions")(0);

  BOOST_REQUIRE(fabs(testY1 - testY2) < delta);
}

/**
 * Check that model can saved / loaded and used. Ensuring that results are the
 * same.
 */
BOOST_AUTO_TEST_CASE(LRModelReload)
{
  constexpr double delta = 1e-5;
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D, N);

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", testX);

  mlpackMain();

  LinearRegression* model = CLI::GetParam<LinearRegression*>("output_model");
  const arma::rowvec testY1 = CLI::GetParam<arma::rowvec>("output_predictions");

  ResetSettings();

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  mlpackMain();

  const arma::rowvec testY2 = CLI::GetParam<arma::rowvec>("output_predictions");

  double norm = arma::norm(testY1 - testY2, 2);
  BOOST_REQUIRE(norm < delta);
}

/**
 * Ensuring that response size is checked.
 */
BOOST_AUTO_TEST_CASE(LRWrongResponseSizeTest)
{
  constexpr int N = 10;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N + 3); // Wrong size.

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that test data dimensionality is checked.
 */
BOOST_AUTO_TEST_CASE(LRWrongDimOfDataTest1)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D - 1, M); // Wrong dimensionality.

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", std::move(testX));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
BOOST_AUTO_TEST_CASE(LRWrongDimOfDataTest2)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));

  mlpackMain();

  LinearRegression* model = CLI::GetParam<LinearRegression*>("output_model");

  ResetSettings();

  arma::mat testX = arma::randu<arma::mat>(D - 1, M); // Wrong dimensionality.
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
BOOST_AUTO_TEST_CASE(LRPredictionSizeCheck)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", std::move(testX));

  mlpackMain();

  const arma::rowvec testY = CLI::GetParam<arma::rowvec>("output_predictions");

  BOOST_REQUIRE_EQUAL(testY.n_rows, 1);
  BOOST_REQUIRE_EQUAL(testY.n_cols, M);
}

/**
 * Ensuring that absence of responses is checked.
 */
BOOST_AUTO_TEST_CASE(LRNoResponses)
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("training", std::move(trainX));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that absence of training data is checked.
 */
BOOST_AUTO_TEST_CASE(LRNoTrainingData)
{
  constexpr int N = 10;

  arma::rowvec trainY = arma::randu<arma::rowvec>(N);
  SetInputParam("training_responses", std::move(trainY));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
