/**
 * @file tests/main_tests/gmm_probability_test.cpp
 * @author Yashwant Singh
 *
 * Test RUN_BINDING() of gmm_probability_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/gmm_probability_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(GmmProbabilityTestFixture);

// Checking the input and output dimensionality.
TEST_CASE_METHOD(GmmProbabilityTestFixture, "GmmProbabilityDimensionality",
                 "[GmmProbabilityMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  GMM gmm(1, 5);
  gmm.Train(std::move(inputData), 5);

  arma::mat inputPoints(5, 5, arma::fill::randu);

  SetInputParam("input", std::move(inputPoints));
  SetInputParam("input_model", &gmm);

  RUN_BINDING();

  REQUIRE(params.Get<arma::mat>("output").n_cols == 5);
  REQUIRE(params.Get<arma::mat>("output").n_rows == 1);

  // Avoid double free (the fixture will try to delete the input model).
  params.Get<GMM*>("input_model") = NULL;
}
