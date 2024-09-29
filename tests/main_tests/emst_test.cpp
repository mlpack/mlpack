/**
 * @file tests/main_tests/emst_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of emst_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/emst/emst_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(EMSTTestFixture);

/**
 * Make sure that Output has 3 Dimensions and
 * check the number of output edges.
 */
TEST_CASE_METHOD(EMSTTestFixture, "EMSTOutputDimensionTest",
                 "[EMSTMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) 2);

  RUN_BINDING();

  // Now check that the output has 3 dimensions.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 3);
  // Check number of output points.
  REQUIRE(params.Get<arma::mat>("output").n_cols == 999);
}

/**
 * Check Naive algorithm Output has 3 Dimensions and
 * check the number of output edges.
 */
TEST_CASE_METHOD(EMSTTestFixture, "EMSTNaiveOutputDimensionTest",
                 "[EMSTMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("naive", true);

  RUN_BINDING();

  // Now check that the output has 3 dimensions.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 3);
  // Check number of output points.
  REQUIRE(params.Get<arma::mat>("output").n_cols == 999);
}

/**
 * Ensure that we can't specify an invalid leaf size.
 */
TEST_CASE_METHOD(EMSTTestFixture, "EMSTInvalidLeafSizeTest",
                 "[EMSTMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) -1); // Invalid leaf size.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that all elements of first two output rows are close to integers.
 */
TEST_CASE_METHOD(EMSTTestFixture, "EMSTFirstTwoOutputRowsIntegerTest",
                 "[EMSTMainTest][BindingTests]")
{
  arma::mat x;
  if (!data::Load("test_data_3_1000.csv", x))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) 2);

  for (size_t i = 0; i < params.Get<arma::mat>("output").n_cols; ++i)
  {
    REQUIRE(params.Get<arma::mat>("output")(0, i) ==
        Approx(std::round(params.Get<arma::mat>("output")(0, i))).
        epsilon(1e-7));
    REQUIRE(params.Get<arma::mat>("output")(1, i) ==
        Approx(std::round(params.Get<arma::mat>("output")(1, i))).
        epsilon(1e-7));
  }
}
