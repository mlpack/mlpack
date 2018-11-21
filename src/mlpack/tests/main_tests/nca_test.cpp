/**
 * @file nca_test.cpp
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
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct NCATestFixture
{
 public:
  NCATestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~NCATestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(NCAMainTest, NCATestFixture);

/**
 * Ensure that, when labels are implicitily given with input,
 * the last column is treated as labels and that we get the
 * desired shape of output.
 */
BOOST_AUTO_TEST_CASE(NCAExplicitImplicitLabelsTest)
{
  // Random dataset.
  arma::mat x;
  x.randu(3, 6);

  SetInputParam("input", std::move(x));

  mlpackMain();

  // Check that last row was treated as label by checking that
  // the output has 1 less row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 2);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 2);

  // Reset Settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Now check that when labels are explicitely given, the last column
  // of input is not treated as labels.
  arma::mat y              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 2);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 2);
}

/**
 * Ensure that when we pass optimizer of type lbfgs, we also get the desired
 * shape of output.
 */
BOOST_AUTO_TEST_CASE(NCALBFGSTest)
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
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 3);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 3);
}

/**
 * Ensure that if labels are of a different size than required
 * by the input, an error occurs.
 */
BOOST_AUTO_TEST_CASE(NCALabelSizeTest)
{
  // Input labels of wrong size.
  arma::mat x               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    1    2 ";

  SetInputParam("input", std::move(x));
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
BOOST_AUTO_TEST_CASE(NCANormalizationTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load vc2_labels.txt!");

  // Set parameters and set normalize to true.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset rettings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  arma::mat inputData2;
  if (!data::Load("vc2.csv", inputData2))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels2;
  if (!data::Load("vc2_labels.txt", labels2))
    BOOST_FAIL("Cannot load vc2_labels.txt!");

  // Use the same input but set normalize to false.
  SetInputParam("input", std::move(inputData2));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("normalize", true);
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is different when step_size is different.
 */
BOOST_AUTO_TEST_CASE(NCADifferentStepSizeTest)
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

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
  BOOST_REQUIRE_GT(arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is different when the tolerance is different.
 */
BOOST_AUTO_TEST_CASE(NCADifferentToleranceTest)
{
  // Random dataset.
  arma::mat x;
  x.randu(3, 600);

  arma::mat y = x;

  // Set parameters with a small tolerance.
  SetInputParam("input", std::move(x));
  SetInputParam("optimizer", std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 0);
  SetInputParam("tolerance", (double) 0.00005);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger tolerance.
  SetInputParam("input", std::move(y));
  SetInputParam("optimizer", std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 0);
  SetInputParam("tolerance", (double) 0.003);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is different when batch_size is different.
 */
BOOST_AUTO_TEST_CASE(NCADifferentBatchSizeTest)
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

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
  BOOST_REQUIRE_GT(arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is different when setting linear_scan to false.
 */
BOOST_AUTO_TEST_CASE(NCALinearScanTest)
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

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

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
  BOOST_REQUIRE_GT(arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that output is the same when same data is used with linear_scan set.
 */
BOOST_AUTO_TEST_CASE(NCALinearScanTest2)
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

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset Settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set same parameter using the same data.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  SetInputParam("input", std::move(y));
  SetInputParam("labels", labels2);
  SetInputParam("linear_scan", true);
  mlpackMain();

  // Check that the output matrices are equal.
  CheckMatrices(output, CLI::GetParam<arma::mat>("output"));
}

/**
 * Ensure that different value of NumBasis results in a
 * different output matrix.
 */
BOOST_AUTO_TEST_CASE(NCADifferentNumBasisTest)
{
  // Simple dataset.
  arma::mat x;
  x.randu(3, 600);

  arma::mat y = x;

  // Set parameters and use a larger num_basis.
  SetInputParam("input", std::move(x));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_basis", (int) 50);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset Settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters with a smaller num_basis.
  SetInputParam("input", std::move(y));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_basis", (int) 10);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

/**
 * Ensure that using a different value of max_iteration
 * results in a different output matrix.
 */
BOOST_AUTO_TEST_CASE(NCADifferentMaxIterationTest)
{
  // Random dataset.
  arma::mat x;
  x.randu(3, 600);

  arma::mat y = x;

  // Set parameters with a small max_iterations.
  SetInputParam("input", std::move(x));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 20);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger max_iterations.
  SetInputParam("input", std::move(y));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 500);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_GT(arma::accu(CLI::GetParam<arma::mat>("output") != output), 0);
}

BOOST_AUTO_TEST_SUITE_END();
