/**
 * @file tests/main_tests/softmax_regression_train_test.cpp
 * @author Manish Kumar
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of softmax_regression_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(SoftmaxRegressionTrainTestFixture);

/**
 * Ensure that we get desired dimensions when both training
 * data and labels are passed.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainOutputDimensionTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  constexpr int M = 5;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));
  arma::mat testX = arma::trans(arma::randu<arma::mat>(M, D));
  size_t testSize = testX.n_cols;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training the model.
  RUN_BINDING();

  // Get the output predictions of the test data.
  arma::Row<size_t> preds;
  params.Get<SoftmaxRegression<>*>("output_model")->Classify(
    std::move(testX), preds);

  // Output predictions size must match the test data set size.
  REQUIRE(preds.n_rows == 1);
  REQUIRE(preds.n_cols == testSize);
}

/**
 * Ensure that labels are necessarily passed when training.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainLabelsLessDimensionTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));

  // Input training data.
  SetInputParam("training", std::move(trainX));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that max_iterations is always non-negative.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainMaxItrTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));

  // Input training data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that lambda is always non-negative.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainLambdaTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));

  // Input training data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("lambda", (double) -0.1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that number_of_classes is always positive.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainNumClassesTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));

  // Input training data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("number_of_classes", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that output object parameters are
 * different for different lambda values.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainDiffLambdaTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));

  // Input training data.
  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  // Train SR for lambda 0.1.
  SetInputParam("lambda", (double) 0.1);

  RUN_BINDING();

  // Store output parameters.
  arma::mat params1 =
    params.Get<SoftmaxRegression<>*>("output_model")->Parameters();

  // Reset passed parameters.
  CleanMemory();
  ResetSettings();


  // Input training data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  // Train SR for lamda 0.9.
  SetInputParam("lambda", (double) 0.9);

  RUN_BINDING();

  arma::mat params2 =
    params.Get<SoftmaxRegression<>*>("output_model")->Parameters();

  REQUIRE(!arma::approx_equal(params1, params2, "absdiff", 1e-7));
}

/**
 * Check that output object parameters are different for different numbers of
 * max_iterations.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainDiffMaxItrTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));

  // Input training data.
  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("max_iterations", (int) 10);

  RUN_BINDING();

  // Store output parameters.
  arma::mat params1 =
    params.Get<SoftmaxRegression<>*>("output_model")->Parameters();

  // Reset passed parameters.
  CleanMemory();
  ResetSettings();

  // Input training data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", (int) 1000);

  RUN_BINDING();

  arma::mat params2 =
    params.Get<SoftmaxRegression<>*>("output_model")->Parameters();

  REQUIRE(!arma::approx_equal(params1, params2, "absdiff", 1e-7));
}

/**
 * Check that output object parameter for no_intercept
 * term is one less than with intercept term.
 */
TEST_CASE_METHOD(SoftmaxRegressionTrainTestFixture,
                 "SoftmaxRegressionTrainDiffInterceptTest",
                 "[SoftmaxRegressionTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY =
    arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));

  // Input training data.
  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("no_intercept", (bool) true);

  RUN_BINDING();

  // Store output parameters.
  arma::mat modparams1 =
    params.Get<SoftmaxRegression<>*>("output_model")->Parameters();

  // Reset passed parameters.
  CleanMemory();
  ResetSettings();

  // Input training data.
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  // no_intercept false by default as well
  SetInputParam("no_intercept", (bool) false);

  RUN_BINDING();

  // Store output parameters.
  arma::mat modparams2 =
    params.Get<SoftmaxRegression<>*>("output_model")->Parameters();

  // Check that initial parameters matrix has one fewer parameter than
  // final parameters matrix.
  REQUIRE(modparams1.n_cols + 1 == modparams2.n_cols);
}
