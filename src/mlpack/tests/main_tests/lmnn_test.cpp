/**
 * @file lmnn_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of lmnn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "LMNN";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include "test_helper.hpp"
#include <mlpack/methods/lmnn/lmnn_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LMNNTestFixture
{
 public:
  LMNNTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LMNNTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LMNNMainTest, LMNNTestFixture);

/**
 * Ensure that, when labels are implicitily given with input,
 * the last column is treated as labels and that we get the
 * desired shape of output.
 */
BOOST_AUTO_TEST_CASE(LMNNExplicitImplicitLabelsTest)
{
  // Dataset containing labels as last column.
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  SetInputParam("input", inputData);

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows,
      inputData.n_rows - 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols,
      inputData.n_rows - 1);

  // Reset Settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Now check that when labels are explicitely given, the last column
  // of input is not treated as labels.
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  SetInputParam("input", inputData);
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows,
      inputData.n_rows);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols,
      inputData.n_rows);
}

/**
 * Ensure that when we pass optimizer of type lbfgs, we also get the desired
 * shape of output.
 */
BOOST_AUTO_TEST_CASE(LMNNOptimizerTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("bbsgd"));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows,
      inputData.n_rows);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols,
      inputData.n_rows);

  // Reset rettings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("sgd"));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows,
      inputData.n_rows);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols,
      inputData.n_rows);

  // Reset rettings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Input random data points.
  SetInputParam("input", inputData);
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows,
      inputData.n_rows);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols,
      inputData.n_rows);
}

/**
 * Ensure that if labels are of a different size than required
 * by the input, an error occurs.
 */
BOOST_AUTO_TEST_CASE(LMNNLabelSizeTest)
{
  // Input labels of wrong size.
  arma::mat inputData      = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    0    0    0";

  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));

  // Check that an error is thrown.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that setting normalize as true results in a
 * different output matrix then when set to false.
 */
BOOST_AUTO_TEST_CASE(LMNNNormalizationTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters and set normalize to true.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset rettings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Use the same input but set normalize to false.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("normalize", true);
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is different when step_size is different.
 */
BOOST_AUTO_TEST_CASE(LMNNDifferentStepSizeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small step_size.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("step_size", (double) 0.01);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger step_size.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("step_size", (double) 20.5);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is different when the tolerance is different.
 */
BOOST_AUTO_TEST_CASE(LMNNDifferentToleranceTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small tolerance.
  SetInputParam("input", inputData);
  SetInputParam("tolerance", (double) 0.00005);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger tolerance.
  SetInputParam("input", std::move(inputData));
  SetInputParam("tolerance", (double) 0.3);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is different when batch_size is different.
 */
BOOST_AUTO_TEST_CASE(LMNNDifferentBatchSizeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small batch_size.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("batch_size", (int) 20);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger batch_size.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("batch_size", (int) 30);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that different value of number of targets results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNNumTargetsTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_targets", std::move(1));

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("num_targets", std::move(5));

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that different value of regularization results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNRegularizationTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("regularization", std::move(1.0));

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("regularization", std::move(0.1));

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that different value of range results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNRangeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("range", std::move(50));

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that different value of batch delta results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNBatchDeltaTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("bbsgd"));

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("bbsgd"));
  SetInputParam("batch_delta", std::move(10.0));

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that different value of epsilon results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNEpsilonTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("amsgrad"));

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("amsgrad"));
  SetInputParam("epsilon", std::move(10.0));

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that different value of wolfe results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNWolfeTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_targets", std::move(5));
  SetInputParam("wolfe", 0.1);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set different parameters.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_targets", std::move(5));
  SetInputParam("wolfe", 1.0);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that different value of NumBasis results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNDifferentNumBasisTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters and use a larger num_basis.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_targets", std::move(5));
  SetInputParam("num_basis", (int) 100);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset Settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters with a smaller num_basis.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_targets", std::move(5));
  SetInputParam("num_basis", (int) 1);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that using a different value of max_iteration
 * results in a different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNDifferentMaxIterationTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small max_iterations.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_targets", std::move(5));
  SetInputParam("max_iterations", (int) 2);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger max_iterations.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", labels);
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_targets", std::move(5));
  SetInputParam("max_iterations", (int) 500);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that using a different value of passes
 * results in a different output matrix.
 */
BOOST_AUTO_TEST_CASE(LMNNDifferentPassesTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Set parameters with a small max_iterations.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("passes", (int) 2);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger max_iterations.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", labels);
  SetInputParam("passes", (int) 6);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(
      arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that number of targets, range, batch size must be always positive
 * and regularization, step size, max iterations, passes & tolerance are
 * always non-negative 
 */
BOOST_AUTO_TEST_CASE(LMNNBoundsTest)
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    BOOST_FAIL("Cannot load iris.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    BOOST_FAIL("Cannot load iris_labels.txt!");

  // Test for number of targets value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("num_targets", (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Test for range value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("range", (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Test for batch size value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("batch_size", (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Test for regularization value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("regularization", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Test for step size value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("step_size", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Test for max iterations value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("max_iterations", (int) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Test for passes value.

  // Input training data.
  SetInputParam("input", inputData);
  SetInputParam("labels", labels);
  SetInputParam("passes", (int) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Test for tolerance value.

  // Input training data.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("tolerance", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
