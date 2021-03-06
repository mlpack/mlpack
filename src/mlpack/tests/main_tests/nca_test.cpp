/**
 * @file tests/main_tests/nca_test.cpp
 * @author Yasmine Dumouchel
 *
 * Test mlpackMain() of nca_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "nca";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/nca/nca_main.cpp>

#include <ensmallen.hpp>

#include "test_helper.hpp"
#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

struct NCATestFixture
{
 public:
  NCATestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~NCATestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/**
 * Ensure that, when labels are implicitily given with input,
 * the last column is treated as labels and that we get the
 * desired shape of output.
 */
TEST_CASE_METHOD(NCATestFixture, "NCAExplicitImplicitLabelsTest",
                "[NCAMainTest][BindingTests]")
{
  // Random dataset.
  arma::mat x;
  x.randu(3, 6);

  SetInputParam("input", std::move(x));

  mlpackMain();

  // Check that last row was treated as label by checking that
  // the output has 1 less row.
  REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 2);
  REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 2);

  // Reset Settings.
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Now check that when labels are explicitely given, the last column
  // of input is not treated as labels.
  arma::mat y              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 2);
  REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 2);
}

/**
 * Ensure that when we pass optimizer of type lbfgs, we also get the desired
 * shape of output.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALBFGSTest",
                "[NCAMainTest][BindingTests]")
{
  arma::mat x;
  x.randu(3, 100);
  arma::Row<size_t> labels;
  labels.zeros(100);

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 3);
  REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 3);
}

/**
 * Ensure that if labels are of a different size than required
 * by the input, an error occurs.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALabelSizeTest",
                "[NCAMainTest][BindingTests]")
{
  // Input labels of wrong size.
  arma::mat x               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    1    2 ";

  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));

  // Check that an error is thrown.
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that setting normalize as true results in a
 * different output matrix then when set to false.
 */
TEST_CASE_METHOD(NCATestFixture, "NCANormalizationTest",
                "[NCAMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load vc2_labels.txt!");

  // Set parameters and set normalize to true.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  mlpackMain();

  arma::mat output = IO::GetParam<arma::mat>("output");

  // Reset rettings.
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  arma::mat inputData2;
  if (!data::Load("vc2.csv", inputData2))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels2;
  if (!data::Load("vc2_labels.txt", labels2))
    FAIL("Cannot load vc2_labels.txt!");

  // Use the same input but set normalize to false.
  SetInputParam("input", std::move(inputData2));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("normalize", true);
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  mlpackMain();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(IO::GetParam<arma::mat>("output") != output) > 0);
}

/**
 * Ensure that output is different when step_size is different.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentStepSizeTest",
                "[NCAMainTest][BindingTests]")
{
  // Simple dataset with 6 points and two classes.
  arma::mat x              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Set parameters with a small step_size.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("step_size", (double) 1.2);
  SetInputParam("linear_scan", true);

  mlpackMain();

  arma::mat output = IO::GetParam<arma::mat>("output");

  // Reset settings.
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Same dataset.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Set parameters using the same input but with a larger step_size.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("step_size", (double) 20.5);
  SetInputParam("linear_scan", true);

  mlpackMain();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(IO::GetParam<arma::mat>("output") != output) > 0);
}

/**
 * Ensure that output is different when the tolerance is different.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentToleranceTest",
                "[NCAMainTest][BindingTests]")
{
  // We aren't guaranteed that the test will be successful, so we run it
  // multiple times.
  bool success = false;
  size_t trial = 0;
  while (trial < 5)
  {
    // Random dataset.
    arma::mat x;
    x.randu(3, 600);
    arma::Row<size_t> labels = arma::randi<arma::Row<size_t>>(600,
        arma::distr_param(0, 1));

    arma::mat y = x;
    arma::Row<size_t> labels2 = labels;

    // Set parameters with a small tolerance.
    SetInputParam("input", std::move(x));
    SetInputParam("labels", std::move(labels));
    SetInputParam("optimizer", std::string("lbfgs"));
    SetInputParam("max_iterations", (int) 0);
    SetInputParam("tolerance", (double) 1e-8);

    mlpackMain();

    arma::mat output = IO::GetParam<arma::mat>("output");

    // Reset settings.
    IO::ClearSettings();
    IO::RestoreSettings(testName);

    // Set parameters using the same input but with a larger tolerance.
    SetInputParam("input", std::move(y));
    SetInputParam("labels", std::move(labels2));
    SetInputParam("optimizer", std::string("lbfgs"));
    SetInputParam("max_iterations", (int) 0);
    SetInputParam("tolerance", (double) 100.0);

    mlpackMain();

    // Check that the output matrices are different.
    success = (arma::accu(IO::GetParam<arma::mat>("output") != output) > 0);
    if (success)
      break;

    ++trial;
  }

  REQUIRE(success == true);
}

/**
 * Ensure that output is different when batch_size is different.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentBatchSizeTest",
                "[NCAMainTest][BindingTests]")
{
  // Simple dataset with 6 points and two classes.
  arma::mat x              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Set parameters with a small batch_size.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("batch_size", (int) 2);
  SetInputParam("linear_scan", true);

  mlpackMain();

  arma::mat output = IO::GetParam<arma::mat>("output");

  // Reset settings.
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Input the same dataset.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Set parameters using the same input but with a larger batch_size.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("batch_size", (int) 3);
  SetInputParam("linear_scan", true);

  mlpackMain();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(IO::GetParam<arma::mat>("output") != output) > 0);
}

/**
 * Ensure that output is different when setting linear_scan to false.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALinearScanTest",
                "[NCAMainTest][BindingTests]")
{
  // Simple dataset with 6 points and two classes.
  arma::mat x               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels  = " 0    0    0    1    1    1   ";

  // Set parameters.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", labels);
  SetInputParam("optimizer", std::string("sgd"));

  mlpackMain();

  arma::mat output = IO::GetParam<arma::mat>("output");

  // Reset settings.
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Input the same dataset.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Set parameters using the same input but set linear_scan flag to false.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", labels2);
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("linear_scan", false);

  mlpackMain();

  // Check that the output matrices are different.
  REQUIRE(arma::accu(IO::GetParam<arma::mat>("output") != output) > 0);
}

/**
 * Ensure that output is the same when same data is used with linear_scan set.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALinearScanTest2",
                "[NCAMainTest][BindingTests]")
{
  // Simple dataset with 6 points and two classes.
  arma::mat x               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels  = " 0    0    0    1    1    1   ";

  // Set same parameter with same data.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", labels);
  SetInputParam("linear_scan", true);

  mlpackMain();

  arma::mat output = IO::GetParam<arma::mat>("output");

  // Reset Settings.
  IO::ClearSettings();
  IO::RestoreSettings(testName);

  // Set same parameter using the same data.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  SetInputParam("input", std::move(y));
  SetInputParam("labels", labels2);
  SetInputParam("linear_scan", true);
  mlpackMain();

  // Check that the output matrices are equal.
  CheckMatrices(output, IO::GetParam<arma::mat>("output"));
}

/**
 * Ensure that different value of NumBasis results in a
 * different output matrix.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentNumBasisTest",
                "[NCAMainTest][BindingTests]")
{
  // This test can randomly fail and it can be okay, so we run multiple times if
  // necessary.
  bool success = false;
  size_t trial = 0;
  while (trial < 5)
  {
    // Simple dataset.
    arma::mat x;
    x.randu(8, 600);
    arma::Row<size_t> labels = arma::randi<arma::Row<size_t>>(600,
        arma::distr_param(0, 1));

    arma::mat y = x;
    arma::Row<size_t> labels2 = labels;

    // Set parameters and use a larger num_basis.
    SetInputParam("input", std::move(x));
    SetInputParam("labels", std::move(labels));
    SetInputParam("optimizer",  std::string("lbfgs"));
    SetInputParam("num_basis", (int) 5);

    mlpackMain();

    arma::mat output = IO::GetParam<arma::mat>("output");

    // Reset Settings.
    IO::ClearSettings();
    IO::RestoreSettings(testName);

    // Set parameters with a smaller num_basis.
    SetInputParam("input", std::move(y));
    SetInputParam("labels", std::move(labels2));
    SetInputParam("optimizer",  std::string("lbfgs"));
    SetInputParam("num_basis", (int) 1);

    mlpackMain();

    // Check that the output matrices are different.
    success = (arma::accu(IO::GetParam<arma::mat>("output") != output) > 0);
    if (success)
      break;

    ++trial;
  }

  REQUIRE(success == true);
}

/**
 * Ensure that using a different value of max_iteration
 * results in a different output matrix.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentMaxIterationTest",
                "[NCAMainTest][BindingTests]")
{
  // This test can randomly fail and it can be okay, so we run multiple times if
  // necessary.
  bool success = false;
  size_t trial = 0;
  while (trial < 5)
  {
    // Random dataset.
    arma::mat x;
    x.randu(3, 600);
    arma::Row<size_t> labels = arma::randi<arma::Row<size_t>>(600,
        arma::distr_param(0, 1));

    arma::mat y = x;
    arma::Row<size_t> labels2 = labels;

    // Set parameters with a small max_iterations.
    SetInputParam("input", std::move(x));
    SetInputParam("labels", std::move(labels));
    SetInputParam("optimizer",  std::string("lbfgs"));
    SetInputParam("max_iterations", (int) 3);

    mlpackMain();

    arma::mat output = IO::GetParam<arma::mat>("output");

    // Reset settings.
    IO::ClearSettings();
    IO::RestoreSettings(testName);

    // Set parameters using the same input but with a larger max_iterations.
    SetInputParam("input", std::move(y));
    SetInputParam("labels", std::move(labels2));
    SetInputParam("optimizer",  std::string("lbfgs"));
    SetInputParam("max_iterations", (int) 500);

    mlpackMain();

    // Check that the output matrices are different.
    success = (arma::accu(IO::GetParam<arma::mat>("output") != output) > 0);
    if (success)
      break;

    ++trial;
  }

  REQUIRE(success == true);
}
