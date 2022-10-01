/**
 * @file tests/main_tests/lmnn_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of lmnn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/lmnn/lmnn_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LMNNTestFixture);

/**
 * Ensure that, when labels are implicitily given with input,
 * the last column is treated as labels and that we get the
 * desired shape of output.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNExplicitImplicitLabelsTest",
                "[LMNNMainTest][BindingTests]")
{
  // Dataset containing labels as last column.
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load iris.csv!");

  SetInputParam("input", inputData);

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows - 1);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows - 1);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows - 1);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);

  // Reset Settings.
  ResetSettings();

  // Now check that when labels are explicitely given, the last column
  // of input is not treated as labels.
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  SetInputParam("input", inputData);
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);
}

/**
 * Ensure that when we pass optimizer of type lbfgs, we also get the desired
 * shape of output.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNOptimizerTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  // TODO: set back to bbsgd---this was done for #1490 and should be reverted
  // when that is fixed.
  SetInputParam("optimizer",  std::string("amsgrad"));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);

  // Reset rettings.
  ResetSettings();

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("sgd"));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);

  // Reset settings.
  ResetSettings();

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);
}

/**
 * Ensure that when we pass a valid initial learning point, we get
 * output of the same dimensions.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNValidDistanceTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Initial learning point.
  arma::mat distance;
  distance.randu(inputData.n_rows - 1, inputData.n_rows);

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", std::move(labels));
  SetInputParam("distance", std::move(distance));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows - 1);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows - 1);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);
}

/**
 * Ensure that when we pass a valid initial square matrix as the learning
 * point, we get output of the same dimensions.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNValidDistanceTest2",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Initial learning point (square matrix).
  arma::mat distance;
  distance.randu(inputData.n_rows, inputData.n_rows);

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", std::move(labels));
  SetInputParam("distance", std::move(distance));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);
}

/**
 * Ensure that when we pass an invalid initial learning point, we get
 * output as the square matrix.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNInvalidDistanceTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Initial learning point.
  arma::mat distance;
  distance.randu(inputData.n_rows + 1, inputData.n_rows);

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", std::move(labels));
  SetInputParam("distance", std::move(distance));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("output").n_cols ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_rows ==
      inputData.n_rows);
  REQUIRE(params.Get<arma::mat>("transformed_data").n_cols ==
      inputData.n_cols);
}

/**
 * Ensure that if number of available labels in a class is less than
 * the number of targets, an error occurs.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNNumTargetsTest",
                "[LMNNMainTest][BindingTests]")
{
  // Input Dataset
  arma::mat inputData      = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1";

  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("k", (int) 5);

  // Check that an error is thrown.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that setting normalize as true results in a
 * different output matrix then when set to false.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffNormalizationTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters and set normalize to true.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Use the same input but set normalize to false.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("normalize", true);
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that output is different when step_size is different.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffStepSizeTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small step_size.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("step_size", (double) 0.01);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set parameters using the same input but with a larger step_size.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("step_size", (double) 20.5);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that output is different when the tolerance is different.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffToleranceTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small tolerance.
  SetInputParam("input", inputData);
  SetInputParam("tolerance", (double) 1e-6);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set parameters using the same input but with a larger tolerance.
  SetInputParam("input", std::move(inputData));
  SetInputParam("tolerance", (double) 0.3);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that output is different when batch_size is different.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffBatchSizeTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small batch_size.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("batch_size", (int) 20);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set parameters using the same input but with a larger batch_size.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("batch_size", (int) 30);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that different value of number of targets results in a
 * different output matrix.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffNumTargetsTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("k", 1);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("k", 5);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that different value of regularization results in a
 * different output matrix.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffRegularizationTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("linear_scan",  (bool) true);
  SetInputParam("regularization", 1.0);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("linear_scan",  (bool) true);
  SetInputParam("regularization", 0.1);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that different value of range results in a
 * different output matrix.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffRangeTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("linear_scan",  (bool) true);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("linear_scan",  (bool) true);
  SetInputParam("range", 100);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that using a different value of max_iteration
 * results in a different output matrix.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffMaxIterationTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small max_iterations.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("linear_scan",  (bool) true);
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("k", 5);
  SetInputParam("max_iterations", (int) 2);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set parameters using the same input but with a larger max_iterations.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", labels);
  SetInputParam("linear_scan",  (bool) true);
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("k", 5);
  SetInputParam("max_iterations", (int) 500);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that using a different value of passes
 * results in a different output matrix.
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNDiffPassesTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small passes.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("linear_scan",  (bool) true);
  SetInputParam("passes", (int) 2);

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");
  arma::mat transformedData = params.Get<arma::mat>("transformed_data");

  // Reset settings.
  ResetSettings();

  // Set parameters using the same input but with a larger passes.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", labels);
  SetInputParam("linear_scan",  (bool) true);
  SetInputParam("passes", (int) 6);

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(params.Get<arma::mat>("output") != output) > 0);
  REQUIRE(arma::accu(params.Get<arma::mat>("transformed_data") !=
      transformedData) > 0);
}

/**
 * Ensure that number of targets, range, batch size must be always positive
 * and regularization, step size, max iterations, rank, passes & tolerance are
 * always non-negative 
 */
TEST_CASE_METHOD(LMNNTestFixture, "LMNNBoundsTest",
                "[LMNNMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load iris_labels.txt!");

  // Test for number of targets value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("k", (int) 0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for range value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("range", (int) 0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for batch size value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("batch_size", (int) 0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for regularization value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("regularization", (double) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for step size value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("step_size", (double) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for max iterations value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("max_iterations", (int) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for passes value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("passes", (int) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for max iterations value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("rank", (int) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Reset settings.
  ResetSettings();

  // Test for tolerance value.

  // Input training data.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("tolerance", (double) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
