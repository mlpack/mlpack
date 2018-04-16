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
#include "test_helper.hpp"
#include <mlpack/methods/nca/nca_main.cpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

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
  arma::mat x;
  x.randu(3, 100);

  // Input random data points.
  SetInputParam("input", std::move(x));

  mlpackMain();

  // Check that last row was treated as label by checking that
  // the output has 1 less row.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 2);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 2);

  // Reset Settings
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Now check that when labels are explicitely given, the last column
  // is not treated as labels.
  arma::mat y;
  y.randu(3, 100);
  arma::Row<size_t> labels;
  labels.zeros(3);

  // Random dataset and labels.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels));

  mlpackMain();

  // Check that final output has expected number of rows and colums.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 3);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 3);
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
  labels.zeros(3);

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
* Ensure that setting normalize as true results in a
* different output matrix then when set to false.
*/
BOOST_AUTO_TEST_CASE(NCANormalizationTest)
{
  // Simple dataset with 6 points and two classes.
  arma::mat x              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Set parameters and set normalize to true.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("normalize", (bool) true);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset rettings.
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Use the same input but set normalize to false.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("normalize", (bool) false);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_EQUAL(arma::approx_equal(
      CLI::GetParam<arma::mat>("output"), output, "absdiff", 0), false);
}

/**
* Ensure that output is different when step_size is different.
*/
BOOST_AUTO_TEST_CASE(NCADiffferentStepSizeTest)
{
  // Simple dataset with 6 points and two classes.
  arma::mat x              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Set parameters with a small step size.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("step_size", (double) 1.2);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset settings
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Same dataset
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Set parameters using the same input but with a larger step size.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("step_size", (double) 20.5);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_EQUAL(arma::approx_equal(
      CLI::GetParam<arma::mat>("output"), output, "absdiff", 0), false);
}

/**
* Ensure that different value of NumBasis results in a
* different output matrix.
*/
BOOST_AUTO_TEST_CASE(NCADiffNumBasisTest)
{
  // Simple dataset with 6 points and two classes.
  arma::mat x;
  x.randu(3, 100);
  arma::Row<size_t> labels;
  labels.zeros(3);

  arma::mat y = x;
  arma::Row<size_t>  labels2 = labels;

  // Set parameters and use a small num_basis
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_basis", (int) 20);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  // Reset Settings
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_basis", (int) 10);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_EQUAL(arma::approx_equal(
      CLI::GetParam<arma::mat>("output"), output, "absdiff", 0), false);
}

/**
* Ensure that using a different value of max_iteration
* results in a different output matrix
*/
BOOST_AUTO_TEST_CASE(NCADiffferentMaxIterationTest)
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

  // Reset settings
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Set parameters using the same input but with a larger max_iterations.
  SetInputParam("input", std::move(y));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 500);

  mlpackMain();

  // Check that the output matrices are different.
  BOOST_REQUIRE_EQUAL(arma::approx_equal(
      CLI::GetParam<arma::mat>("output"), output, "absdiff", 0), false);
}

BOOST_AUTO_TEST_SUITE_END();
