/**
 * @file preprocess_imputer_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of preprocess_imputer_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "PreprocessImputer";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_imputer_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

#include <cmath>

using namespace mlpack;

// Utility function to set a parameter and mark it as passed,
// using copy semantics for lvalues and move semantics for rvalues.
template<typename T>
void SetInputParam(const std::string& name, T&& value)
{
  CLI::GetParam<typename std::remove_reference<T>::type>(name)
       = std::forward<T>(value);
  CLI::SetPassed(name);
}

struct PreprocessImputerTestFixture
{
 public:
  PreprocessImputerTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PreprocessImputerTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessImputerMainTest,
                         PreprocessImputerTestFixture);

/**
 * Check that input and output have same dimensions
 * except for listwise_deletion strategy.
 */
BOOST_AUTO_TEST_CASE(PreprocessImputerDimensionTest)
{
  // Load synthetic dataset.
  arma::mat inputData;
  data::Load("preprocess_imputer_test.csv", inputData);

  // Store size of input dataset.
  size_t input_size  = inputData.n_cols;

  // Input custom data points and labels.
  SetInputParam("input_file", (string) "preprocess_imputer_test.csv");
  SetInputParam("missing_value", (string) "nan");
  SetInputParam("strategy", (string) "mean");
  SetInputParam("output_file", (string) "preprocess_imputer_output_test.csv");

  mlpackMain();

  // Now check that the output has desired dimensions.
  arma::mat outputData;
  data::Load(CLI::GetParam<string>("output_file"), outputData);
  BOOST_REQUIRE_EQUAL(outputData.n_cols, input_size);
}

/**
 * Check that output has less points in
 * case of listwise_deletion strategy.
 */
BOOST_AUTO_TEST_CASE(PreprocessImputerListwiseDimensionTest)
{
  // Load synthetic dataset.
  arma::mat inputData;
  data::Load("preprocess_imputer_test.csv", inputData);

  // Store size of input dataset.
  size_t input_size  = inputData.n_cols;
  size_t countNaN = 0;

  // Count number of unavailable entries.
  for (size_t i = 0; i < input_size; i++)
  {
    if (std::to_string(inputData(1, i)) == "nan")
      countNaN++;
  }

  // Input custom data points and labels.
  SetInputParam("input_file", (string) "preprocess_imputer_test.csv");
  SetInputParam("missing_value", (string) "nan");
  SetInputParam("strategy", (string) "listwise_deletion");
  SetInputParam("output_file", (string) "preprocess_imputer_output_test.csv");

  mlpackMain();

  // Now check that the output has desired dimensions.
  arma::mat outputData;
  data::Load(CLI::GetParam<string>("output_file"), outputData);
  BOOST_REQUIRE_EQUAL(outputData.n_cols + countNaN, input_size);
}

/**
 * Check that invalid strategy can't be specified.
 */
BOOST_AUTO_TEST_CASE(PreprocessImputerStrategyTest)
{
  // Load synthetic dataset.
  arma::mat inputData;
  data::Load("preprocess_imputer_test.csv", inputData);

  // Input custom data points and labels.
  SetInputParam("input_file", (string) "preprocess_imputer_test.csv");
  SetInputParam("missing_value", (string) "nan");
  SetInputParam("strategy", (string) "notmean"); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
