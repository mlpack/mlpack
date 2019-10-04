/**
 * @file linear_svm_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LinearSVMTestFixture
{
 public:
  LinearSVMTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LinearSVMTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LinearSVMMainTest,
                         LinearSVMTestFixture);

/**
 * Ensure that trainingSet are necessarily passed when training.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNoTrainingData)
{
  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("labels", std::move(trainLabels));

  // Training data is not provided. Should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
BOOST_AUTO_TEST_CASE(LinearSVMOutputDimensionTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset iris_test.csv!");

  size_t testSize = testData.n_cols;

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("test", std::move(testData));

  // Training the model.
  mlpackMain();

  // Get the output predictions of the test data.
  const arma::Row<size_t>& testLabels =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Output predictions size must match the test data set size.
  BOOST_REQUIRE_EQUAL(testLabels.n_rows, 1);
  BOOST_REQUIRE_EQUAL(testLabels.n_cols, testSize);
}

/**
 * Ensuring that the labels size is checked.
 */
BOOST_AUTO_TEST_CASE(LinearSVMCheckLabelsSizeTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("vc2_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));

  // Labels with incorrect size. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking two options of specifying labels (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
BOOST_AUTO_TEST_CASE(LinearSVMLabelsRepresentationTest)
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
      std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Both solutions should be equal.
  CheckMatrices(testLabels1, testLabels2);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(LinearSVMModelReuseTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset iris_test.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("test", testData);

  // First solution
  mlpackMain();

  // Get the output model obtained from training.
  LinearSVMModel* model =
      CLI::GetParam<LinearSVMModel*>("output_model");
  // Get the output.
  const arma::Row<size_t>& testLabels1 =
      std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  // Reset the data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  // Second solution.
  mlpackMain();

  // Get the output.
  const arma::Row<size_t>& testLabels2 =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Both solutions should be equal.
  CheckMatrices(testLabels1, testLabels2);
}

/**
 * Checking for dimensionality of the test data set.
 */
BOOST_AUTO_TEST_CASE(LinearSVMCheckDimOfTestData)
{
  // Dimensionality of trainingSet is trainData.n_rows - 1 because labels are
  // not provided.
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset iris_test.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("test", std::move(testData));

  // Dimensionality of test data is wrong. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
BOOST_AUTO_TEST_CASE(LinearSVMCheckDimOfTestData2)
{
  // Dimensionality of trainingSet is trainData.n_rows - 1 because labels are
  // not provided.
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset iris_test.csv!");

  SetInputParam("training", std::move(trainData));

  // Training the model.
  mlpackMain();

  // Get the output model obtained from training.
  LinearSVMModel* model =
      CLI::GetParam<LinearSVMModel*>("output_model");

  // Reset the data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  // Test data dimensionality is wrong. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that max iteration for optimizers is non negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNonNegativeMaxIterationTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("max_iterations", int(-1));

  // Maximum iterations is negative. It should a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that lambda for optimizers is non negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNonNegativeLambdaTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("lambda", double(-0.01));

  // Lambda is negative. It should a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that number of classes for optimizers is non negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNonNegativeNumberOfClassesTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("num_classes", int(-1));

  // Number of classes is negative. It should a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that tolerance is non negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNonNegativeToleranceTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("tolerance", double(-0.01));

  // Tolerance is negative. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that delta is non negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNonNegativeDeltaTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("delta", double(-0.01));

  // Delta is negative. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that epochs is non negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNonNegativeEpochsTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("epochs", int(-1));

  // Epochs is negative. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that number classes must not be zero.
 */
BOOST_AUTO_TEST_CASE(LinearSVMZeroNumberOfClassesTest)
{
  arma::mat trainData = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

  arma::Row<size_t> trainLabels = "0 0 0";

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));

  // Number of classes for optimizer is zero.
  // It should throw a invalid_argument error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that Optimizer must be correct.
 */
BOOST_AUTO_TEST_CASE(LinearSVMOptimizerTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("hello"));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring changing Maximum number of iterations changes the output model.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffMaxIterationsTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("max_iterations", int(1));

  // First solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("max_iterations", int(100));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that lambda has some effects on the output.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffLambdaTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("lambda", double(0.001));

  // First solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("lambda", double(1000));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that delta has some effects on the output.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffDeltaTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);
  SetInputParam("delta", double(1.0));

  // First solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("delta", double(1000));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that no_intercept has some effects on the output.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffInterceptTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", trainData);
  SetInputParam("labels", trainLabels);

  // First solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::mat parameters1 = std::move(
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("no_intercept", bool(true));

  // Second solution.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::mat& parameters2 =
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that no_intercept has some effects on the output
 * when the optimizer is 'psgd'.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffInterceptTestWithPsgd)
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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that step size for optimizer is non negative.
 */
BOOST_AUTO_TEST_CASE(LinearSVMNonNegativeStepSizeTest)
{
  arma::mat trainData;
  if (!data::Load("iris.csv", trainData))
    BOOST_FAIL("Cannot load test dataset iris.csv!");

  arma::Row<size_t> trainLabels;
  if (!data::Load("iris_labels.txt", trainLabels))
    BOOST_FAIL("Cannot load test dataset iris_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(trainLabels));
  SetInputParam("optimizer", std::string("psgd"));
  SetInputParam("step_size", double(-0.01));

  // Step size for optimizer is negative. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that epochs has some effects on the output.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffEpochsTest)
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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that Step size has some effects on the output.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffStepSizeTest)
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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that tolerance has some effects on the output.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffToleranceTest)
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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

/**
 * Ensuring that lbfgs optimizer converges to a different result than psgd.
 */
BOOST_AUTO_TEST_CASE(LinearSVMDiffOptimizerTest)
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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
      CLI::GetParam<LinearSVMModel*>("output_model")->svm.Parameters();

  // Both solutions should be not equal.
  CheckMatricesNotEqual(parameters1, parameters2);
}

BOOST_AUTO_TEST_SUITE_END();

