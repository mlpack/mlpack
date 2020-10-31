/**
 * @file tests/main_tests/linear_svm_test.cpp
 * @author Yashwant Singh Parihar
 *
 * Test mlpackMain() of logistic_regression_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST

static const std::string testName = "LinearSVM";

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm/linear_svm_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

struct LinearSVMTestFixture
{
 public:
  LinearSVMTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~LinearSVMTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  mlpackMain();

  // Get the output predictions of the test data.
  const arma::Row<size_t>& testLabels =
      IO::GetParam<arma::Row<size_t>>("predictions");

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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the output.
  const arma::Row<size_t> testLabels1 =
      std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Now train by providing labels as extra parameter.
  arma::mat trainData2({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}});
  arma::Row<size_t> trainLabels({0, 1, 1});

  SetInputParam("training", std::move(trainData2));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("test", std::move(testData));

  // The second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // get the output
  const arma::Row<size_t>& testLabels2 =
      IO::GetParam<arma::Row<size_t>>("predictions");

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
  mlpackMain();

  // Get the output model obtained from training.
  LinearSVMModel* model =
      IO::GetParam<LinearSVMModel*>("output_model");
  // Get the output.
  const arma::Row<size_t>& testLabels1 =
      std::move(IO::GetParam<arma::Row<size_t>>("predictions"));

  // Reset the data passed.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;
  IO::GetSingleton().Parameters()["test"].wasPassed = false;

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  // Second solution.
  mlpackMain();

  // Get the output.
  const arma::Row<size_t>& testLabels2 =
      IO::GetParam<arma::Row<size_t>>("predictions");

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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  mlpackMain();

  // Get the output model obtained from training.
  LinearSVMModel* model =
      IO::GetParam<LinearSVMModel*>("output_model");

  // Reset the data passed.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;
  IO::GetSingleton().Parameters()["labels"].wasPassed = false;

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  // Test data dimensionality is wrong. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("max_iterations", int(100));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("lambda", double(1000));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("delta", double(1000));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("no_intercept", bool(true));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("no_intercept", bool(true));

  // Second solution.
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("epochs", int(10));

  // Second solution.
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("num_classes", int(2));
  SetInputParam("step_size", double(1.02));

  // Second solution.
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("num_classes", int(2));
  SetInputParam("tolerance", double(1e-10));

  // Second solution.
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

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
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));

  // Second solution.
  #ifdef HAS_OPENMP
  omp_set_num_threads(1);
  #endif

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      IO::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}
