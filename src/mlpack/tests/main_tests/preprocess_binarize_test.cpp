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
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "PreprocessBinarize";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_binarize_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

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
    bindings::tests::CleanMemory();
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
  // Create a synthetic dataset.
  arma::mat inputData = arma::randu<arma::mat>(2, 5);

  // Store size of input dataset.
  size_t inputSize  = inputData.n_cols;

  // Input custom data and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 0.5);
  SetInputParam("dimension", (int) 1);

  mlpackMain();

  // Now check that the output has desired dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 2);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, inputSize);
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

/**
 * Check that binarization took place for the specified dimension.
 */
BOOST_AUTO_TEST_CASE(PreprocessBinarizeVerificationTest)
{
  arma::mat inputData({{7.0, 4.0, 5.0}, {2.0, 5.0, 9.0}, {7.0, 3.0, 8.0}});

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 5.0);
  SetInputParam("dimension", (int) 1);

  mlpackMain();

  arma::mat output;
  output = std::move(CLI::GetParam<arma::mat>("output"));

  // All values dimension should remain unchanged.
  BOOST_REQUIRE_CLOSE(output(0, 0), 7.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 1), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 2), 5.0, 1e-5);

  // All values should be binarized according to the threshold.
  BOOST_REQUIRE_SMALL(output(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(output(1, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 2), 1.0, 1e-5);

  // All values dimension should remain unchanged.
  BOOST_REQUIRE_CLOSE(output(2, 0), 7.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 1), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 2), 8.0, 1e-5);
}

/**
 * Check that all dimensions are binarized when dimension is not specified.
 */
BOOST_AUTO_TEST_CASE(PreprocessBinarizeDimensionLessVerTest)
{
  arma::mat inputData({{7.0, 4.0, 5.0}, {2.0, 5.0, 9.0}, {7.0, 3.0, 8.0}});

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 5.0);

  mlpackMain();

  arma::mat output;
  output = std::move(CLI::GetParam<arma::mat>("output"));

  // All values should be binarized according to the threshold.
  BOOST_REQUIRE_CLOSE(output(0, 0), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(output(0, 1), 1e-5);
  BOOST_REQUIRE_SMALL(output(0, 2), 1e-5);
  BOOST_REQUIRE_SMALL(output(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(output(1, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 2), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 0), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(output(2, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 2), 1.0, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
