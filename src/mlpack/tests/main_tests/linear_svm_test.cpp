/**
 * @file tests/main_tests/linear_svm_test.cpp
 * @author Yashwant Singh Parihar
 *
 * Test RUN_BINDING() of logistic_regression_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm/linear_svm_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LinearSVMTestFixture);

/**
 * Ensure that trainingSet are necessarily passed when training.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMNoTrainingData",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("labels", std::move(trainLabels));

  // Training data is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMOutputDimensionTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    FAIL("Cannot load test dataset iris_test.csv!");

  size_t testSize = testData.n_cols;

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("test", std::move(testData));

  // Training the model.
  RUN_BINDING();

  // Get the output predictions of the test data.
  const arma::Row<size_t>& testLabels =
      params.Get<arma::Row<size_t>>("predictions");

  // Output predictions size must match the test data set size.
  REQUIRE(testLabels.n_rows == 1);
  REQUIRE(testLabels.n_cols == testSize);
}

/**
 * Ensuring that the labels size is checked.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMCheckLabelsSizeTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("vc2_labels.txt", trainLabels))
    FAIL("Cannot load test dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));

  // Labels with incorrect size. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Checking two options of specifying labels (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMLabelsRepresentationTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}, {0, 1, 1}});
  arma::mat testData({{4.0, 5.0}, {1.0, 6.0}});

  SetInputParam("training", std::move(trainData1));
  SetInputParam("test", testData);

  // The first solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the output.
  const arma::Row<size_t> testLabels1 =
      std::move(params.Get<arma::Row<size_t>>("predictions"));

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  // Now train by providing labels as extra parameter.
  arma::mat trainData2({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}});
  arma::Row<size_t> trainLabels({0, 1, 1});

  SetInputParam("training", std::move(trainData2));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("test", std::move(testData));

  // The second solution.
  FixedRandomSeed();
  RUN_BINDING();

  // get the output
  const arma::Row<size_t>& testLabels2 =
      params.Get<arma::Row<size_t>>("predictions");

  // Both solutions should be equal.
  CheckMatrices(testLabels1, testLabels2);
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMModelReuseTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    FAIL("Cannot load test dataset iris_test.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("test", testData);

  // First solution
  RUN_BINDING();

  // Get the output model obtained from training.
  LinearSVMModel* model = params.Get<LinearSVMModel*>("output_model");
  params.Get<LinearSVMModel*>("output_model") = NULL;

  // Get the output.
  arma::Row<size_t> testLabels1 =
      std::move(params.Get<arma::Row<size_t>>("predictions"));

  // Reset the data passed.
  CleanMemory();
  ResetSettings();

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  // Second solution.
  RUN_BINDING();

  // Get the output.
  const arma::Row<size_t>& testLabels2 =
      params.Get<arma::Row<size_t>>("predictions");

  // Both solutions should be equal.
  CheckMatrices(testLabels1, testLabels2);
}

/**
 * Checking for dimensionality of the test data set.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMCheckDimOfTestData",
                 "[LinearSVMMainTest][BindingTests]")
{
  // Dimensionality of trainingSet is trainData.n_rows - 1 because labels are
  // not provided.
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    FAIL("Cannot load test dataset iris_test.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("test", std::move(testData));

  // Dimensionality of test data is wrong. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMCheckDimOfTestData2",
                 "[LinearSVMMainTest][BindingTests]")
{
  // Dimensionality of trainingSet is trainData.n_rows - 1 because labels are
  // not provided.
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    FAIL("Cannot load test dataset iris_test.csv!");

  SetInputParam("training", std::move(trainData));

  // Training the model.
  RUN_BINDING();

  // Get the output model obtained from training.
  LinearSVMModel* model =
      params.Get<LinearSVMModel*>("output_model");

  // Reset the data passed.
  ResetSettings();

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  // Test data dimensionality is wrong. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that max iteration for optimizers is non negative.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMNonNegativeMaxIterationTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("max_iterations", int(-1));

  // Maximum iterations is negative. It should a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that lambda for optimizers is non negative.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMNonNegativeLambdaTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("lambda", double(-0.01));

  // Lambda is negative. It should a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that number of classes for optimizers is non negative.
 */
TEST_CASE_METHOD(LinearSVMTestFixture,
                 "LinearSVMNonNegativeNumberOfClassesTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("num_classes", int(-1));

  // Number of classes is negative. It should a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that tolerance is non negative.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMNonNegativeToleranceTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("tolerance", double(-0.01));

  // Tolerance is negative. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that delta is non negative.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMNonNegativeDeltaTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("delta", double(-0.01));

  // Delta is negative. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that epochs is non negative.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMNonNegativeEpochsTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("epochs", int(-1));

  // Epochs is negative. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that number classes must not be one.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMZeroNumberOfClassesTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

  arma::Row<size_t> trainLabels = "0 0 0";

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));

  // Number of classes for optimizer is only one.
  // It should throw a invalid_argument error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/**
 * Ensuring that Optimizer must be correct.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMOptimizerTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("hello"));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring changing Maximum number of iterations changes the output model.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffMaxIterationsTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("max_iterations", int(1));

  // First solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("max_iterations", int(100));

  // Second solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that lambda has some effects on the output.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffLambdaTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("lambda", double(0.001));

  // First solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("lambda", double(1000));

  // Second solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that delta has some effects on the output.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffDeltaTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("delta", double(1.0));

  // First solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("delta", double(1000));

  // Second solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that no_intercept has some effects on the output.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffInterceptTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);

  // First solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("no_intercept", bool(true));

  // Second solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that no_intercept has some effects on the output
 * when the optimizer is 'psgd'.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffInterceptTestWithPsgd",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

  arma::Row<size_t> trainLabels = "1 0 1";

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("optimizer", std::string("psgd"));

  // First solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("no_intercept", bool(true));

  // Second solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that step size for optimizer is non negative.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMNonNegativeStepSizeTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("step_size", double(-0.01));

  // Step size for optimizer is negative. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that epochs has some effects on the output.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffEpochsTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

  arma::Row<size_t> trainLabels = "1 0 1";

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("epochs", int(5));

  // First solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("epochs", int(10));

  // Second solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that Step size has some effects on the output.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffStepSizeTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

  arma::Row<size_t> trainLabels = "1 0 1";

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("num_classes", int(2));
  SetInputParam("step_size", double(0.02));

  // First solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("num_classes", int(2));
  SetInputParam("step_size", double(1.02));

  // Second solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that tolerance has some effects on the output.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffToleranceTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

  arma::Row<size_t> trainLabels = "1 0 1";

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("num_classes", int(2));
  SetInputParam("tolerance", double(1e-1));

  // First solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("num_classes", int(2));
  SetInputParam("tolerance", double(1e-10));

  // Second solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that lbfgs optimizer converges to a different result than psgd.
 */
TEST_CASE_METHOD(LinearSVMTestFixture, "LinearSVMDiffOptimizerTest",
                 "[LinearSVMMainTest][BindingTests]")
{
  arma::mat trainData = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

  arma::Row<size_t> trainLabels = "1 0 1";

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("optimizer", std::string("lbfgs"));

  // First solution.
  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));

  // Second solution.
  #ifdef MLPACK_USE_OPENMP
  omp_set_num_threads(1);
  #endif

  FixedRandomSeed();
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      params.Get<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}
