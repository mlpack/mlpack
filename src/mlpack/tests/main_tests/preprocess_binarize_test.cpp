/**
 * @file tests/main_tests/preprocess_binarize_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of preprocess_binarize_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/preprocess/preprocess_binarize_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(PreprocessBinarizeTestFixture);

/**
 * Check that input and output have same dimensions.
 */
TEST_CASE_METHOD(
    PreprocessBinarizeTestFixture, "PreprocessBinarizeDimensionTest",
    "[PreprocessBinarizeMainTest][BindingTests]")
{
  // Create a synthetic dataset.
  arma::mat inputData = arma::randu<arma::mat>(2, 5);

  // Store size of input dataset.
  size_t inputSize  = inputData.n_cols;

  // Input custom data and labels.
  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 0.5);
  SetInputParam("dimension", (int) 1);

  RUN_BINDING();

  // Now check that the output has desired dimensions.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 2);
  REQUIRE(params.Get<arma::mat>("output").n_cols == inputSize);
}

/**
 * Check that specified dimension is non-negative.
 */
TEST_CASE_METHOD(
    PreprocessBinarizeTestFixture, "PreprocessBinarizeNegativeDimensionTest",
    "[PreprocessBinarizeMainTest][BindingTests]")
{
  arma::mat inputData = arma::randu<arma::mat>(2, 2);

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 0.5);
  SetInputParam("dimension", (int) -2); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that we can't specify a dimension larger than input.
 */
TEST_CASE_METHOD(
    PreprocessBinarizeTestFixture, "PreprocessBinarizelargerDimensionTest",
    "[PreprocessBinarizeMainTest][BindingTests]")
{
  arma::mat inputData = arma::randu<arma::mat>(2, 2);

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 0.5);
  SetInputParam("dimension", (int) 6); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that binarization took place for the specified dimension.
 */
TEST_CASE_METHOD(
    PreprocessBinarizeTestFixture, "PreprocessBinarizeVerificationTest",
    "[PreprocessBinarizeMainTest][BindingTests]")
{
  arma::mat inputData({{7.0, 4.0, 5.0}, {2.0, 5.0, 9.0}, {7.0, 3.0, 8.0}});

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 5.0);
  SetInputParam("dimension", (int) 1);

  RUN_BINDING();

  arma::mat output;
  output = std::move(params.Get<arma::mat>("output"));

  // All values dimension should remain unchanged.
  REQUIRE(output(0, 0) == Approx(7.0).epsilon(1e-7));
  REQUIRE(output(0, 1) == Approx(4.0).epsilon(1e-7));
  REQUIRE(output(0, 2) == Approx(5.0).epsilon(1e-7));

  // All values should be binarized according to the threshold.
  REQUIRE(output(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(output(1, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(output(1, 2) == Approx(1.0).epsilon(1e-7));

  // All values dimension should remain unchanged.
  REQUIRE(output(2, 0) == Approx(7.0).epsilon(1e-7));
  REQUIRE(output(2, 1) == Approx(3.0).epsilon(1e-7));
  REQUIRE(output(2, 2) == Approx(8.0).epsilon(1e-7));
}

/**
 * Check that all dimensions are binarized when dimension is not specified.
 */
TEST_CASE_METHOD(
    PreprocessBinarizeTestFixture, "PreprocessBinarizeDimensionLessVerTest",
    "[PreprocessBinarizeMainTest][BindingTests]")
{
  arma::mat inputData({{7.0, 4.0, 5.0}, {2.0, 5.0, 9.0}, {7.0, 3.0, 8.0}});

  SetInputParam("input", std::move(inputData));
  SetInputParam("threshold", (double) 5.0);

  RUN_BINDING();

  arma::mat output;
  output = std::move(params.Get<arma::mat>("output"));

  // All values should be binarized according to the threshold.
  REQUIRE(output(0, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(output(0, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(output(0, 2) == Approx(0.0).margin(1e-5));
  REQUIRE(output(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(output(1, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(output(1, 2) == Approx(1.0).epsilon(1e-7));
  REQUIRE(output(2, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(output(2, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(output(2, 2) == Approx(1.0).epsilon(1e-7));
}
