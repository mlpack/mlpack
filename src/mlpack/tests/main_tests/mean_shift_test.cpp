/**
 * @file mean_shift_test.cpp
 * @author Tan Jun An
 *
 * Test mlpackMain() of mean_shift_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "Mean Shift";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/mean_shift/mean_shift_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct MeanShiftTestFixture
{
 public:
  MeanShiftTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~MeanShiftTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(MeanShiftMainTest, MeanShiftTestFixture);

// Generate dataset; written transposed because it's easier to read.
arma::mat meanShiftData("  0.0   0.0;" // Class 1.
                     "  0.3   0.4;"
                     "  0.1   0.0;"
                     "  0.1   0.3;"
                     " -0.2  -0.2;"
                     " -0.1   0.3;"
                     " -0.4   0.1;"
                     "  0.2  -0.1;"
                     "  0.3   0.0;"
                     " -0.3  -0.3;"
                     "  0.1  -0.1;"
                     "  0.2  -0.3;"
                     " -0.3   0.2;"
                     " 10.0  10.0;" // Class 2.
                     " 10.1   9.9;"
                     "  9.9  10.0;"
                     " 10.2   9.7;"
                     " 10.2   9.8;"
                     "  9.7  10.3;"
                     "  9.9  10.1;"
                     "-10.0   5.0;" // Class 3.
                     " -9.8   5.1;"
                     " -9.9   4.9;"
                     "-10.0   4.9;"
                     "-10.2   5.2;"
                     "-10.1   5.1;"
                     "-10.3   5.3;"
                     "-10.0   4.8;"
                     " -9.6   5.0;"
                     " -9.8   5.1;");

/**
 * Ensure that the output has 1 row for the labels and
 * check the number of points remain the same.
 */
BOOST_AUTO_TEST_CASE(MeanShiftOutputDimensionTest)
{
  // Input random data points.
  SetInputParam("input", meanShiftData);

  mlpackMain();

  // Now check that the output has 1 extra row for labels.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 2 + 1);
  // Check number of output points are the same.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 30);
}

/**
 * Ensure that we can't specify an invalid max number of iterations.
 */
BOOST_AUTO_TEST_CASE(MeanShiftInvalidMaxNumberOfIterations)
{
  // Input random data points.
  SetInputParam("input", meanShiftData);
  SetInputParam("max_iterations", (int) -1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();