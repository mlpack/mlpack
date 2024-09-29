/**
 * @file tests/main_tests/gmm_generate_test.cpp
 * @author Yashwant Singh
 *
 * Test RUN_BINDING() of gmm_generate_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/gmm_generate_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(GmmGenerateTestFixture);

// Checking that Samples must greater than 0.
TEST_CASE_METHOD(GmmGenerateTestFixture, "GmmGenerateSamplesTest",
                 "[GmmGenerateMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  GMM gmm(1, 5);
  gmm.Train(inputData, 5);

  SetInputParam("input_model", &gmm);

  SetInputParam("samples", 0); // Invalid
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Avoid double free (the fixture will try to delete the input model).
  params.Get<GMM*>("input_model") = NULL;
}

// Checking dimensionality of output.
TEST_CASE_METHOD(GmmGenerateTestFixture, "GmmGenerateDimensionality",
                 "[GmmGenerateMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  GMM gmm(1, 5);
  gmm.Train(inputData, 5);
  SetInputParam("input_model", &gmm);
  SetInputParam("samples", (int) 10);

  RUN_BINDING();

  arma::mat output = std::move(params.Get<arma::mat>("output"));

  REQUIRE(output.n_rows == gmm.Dimensionality());
  REQUIRE(output.n_cols == (int) 10);

  // Avoid double free (the fixture will try to delete the input model).
  params.Get<GMM*>("input_model") = NULL;
}
