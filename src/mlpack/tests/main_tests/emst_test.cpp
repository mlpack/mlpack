/**
 * @file emst_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of emst_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "EMST";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/emst/emst_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

#include <boost/math/special_functions/round.hpp>

using namespace mlpack;

struct EMSTTestFixture
{
 public:
  EMSTTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~EMSTTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(EMSTMainTest, EMSTTestFixture);

/**
 * Make sure that Output has 3 Dimensions and
 * check the number of output edges.
 */
BOOST_AUTO_TEST_CASE(EMSTOutputDimensionTest)
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) 2);

  mlpackMain();

  // Now check that the output has 3 dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 3);
  // Check number of output points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 999);
}

/**
 * Check Naive algorithm Output has 3 Dimensions and
 * check the number of output edges.
 */
BOOST_AUTO_TEST_CASE(EMSTNaiveOutputDimensionTest)
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("naive", true);

  mlpackMain();

  // Now check that the output has 3 dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 3);
  // Check number of output points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 999);
}

/**
 * Ensure that we can't specify an invalid leaf size.
 */
BOOST_AUTO_TEST_CASE(EMSTInvalidLeafSizeTest)
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) -1); // Invalid leaf size.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that all elements of first two output rows are close to integers.
 */
BOOST_AUTO_TEST_CASE(EMSTFirstTwoOutputRowsIntegerTest)
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) 2);

  for (size_t i = 0; i < CLI::GetParam<arma::mat>("output").n_cols; i++)
  {
    BOOST_REQUIRE_CLOSE(CLI::GetParam<arma::mat>("output")(0, i),
        boost::math::iround(CLI::GetParam<arma::mat>("output")(0, i)), 1e-5);
    BOOST_REQUIRE_CLOSE(CLI::GetParam<arma::mat>("output")(1, i),
        boost::math::iround(CLI::GetParam<arma::mat>("output")(1, i)), 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
