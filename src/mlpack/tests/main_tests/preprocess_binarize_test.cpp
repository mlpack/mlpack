/**
 * @file preprocess_binarize_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of preprocess_binarize_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "PreprocessBinarize";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_binarize_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

// Utility function to set a parameter and mark it as passed,
// using copy semantics for lvalues and move semantics for rvalues.
template<typename T>
void SetInputParam(const std::string& name, T&& value)
{
  CLI::GetParam<typename std::remove_reference<T>::type>(name)
       = std::forward<T>(value);
  CLI::SetPassed(name);
}

struct PreprocessBinarizeTestFixture
{
 public:
  PreprocessBinarizeTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PreprocessBinarizeTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessBinarizeMainTest,
                         PreprocessBinarizeTestFixture);

/**
 * Check that input and output have same dimensions.
 */
BOOST_AUTO_TEST_CASE(PreprocessBinarizeDimensionTest)
{
  // synthetic dataset.
  arma::mat inputData = arma::randu<arma::mat>(2, 5);

  // Store size of input dataset.
  size_t input_size  = inputData.n_cols;

  // Input custom data points and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 0.5);
  SetInputParam("dimension", (int) 1);

  mlpackMain();

  // Now check that the output has desired dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 2);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, input_size);
}

/**
 * Check that specified dimension is non-negative.
 */
BOOST_AUTO_TEST_CASE(PreprocessBinarizeNegativeDimensionTest)
{
  arma::mat inputData = arma::randu<arma::mat>(2, 2);

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 0.5);
  SetInputParam("dimension", (int) -2); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify a dimension larger than input.
 */
BOOST_AUTO_TEST_CASE(PreprocessBinarizelargerDimensionTest)
{
  arma::mat inputData = arma::randu<arma::mat>(2, 2);

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 0.5);
  SetInputParam("dimension", (int) 6); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
