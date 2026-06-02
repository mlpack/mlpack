/**
 * @file tests/main_tests/random_forest_probabilities_test.cpp
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of random_forest_probabilities_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest_probabilities_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(RandomForestProbabilitiesTestFixture);

/**
 * Check that absence of model throws error.
 */
TEST_CASE_METHOD(RandomForestProbabilitiesTestFixture,
                 "RandomForestProbabilitiesNoModelTest",
                 "[RandomForestProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat testX = arma::randu<arma::mat>(N, D);
  SetInputParam("test", std::move(testX));
  // (Required) model is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that untrained model throws error.
 */
TEST_CASE_METHOD(RandomForestProbabilitiesTestFixture,
                 "RandomForestProbabilitiesEmptyModelTest",
                 "[RandomForestProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat testX = arma::randu<arma::mat>(N, D);
  SetInputParam("test", std::move(testX));
  RandomForestModel* model = new RandomForestModel;
  SetInputParam("input_model", std::move(model));
  // (Required) model is not trained. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/**
 * Check that lack of data throws error.
 */
TEST_CASE_METHOD(RandomForestProbabilitiesTestFixture,
                 "RandomForestProbabilitiesNoDataTest",
                 "[RandomForestProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::randu<arma::mat>(N, D);
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  // Initial model.
  RandomForestModel* model = new RandomForestModel;
  // Set as input.
  SetInputParam("input_model", std::move(model));
  // No data, expect error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that probilities size corresponds to test data size.
 */
TEST_CASE_METHOD(RandomForestProbabilitiesTestFixture,
                 "RandomForestProbabilitiesPredictionSizeTest",
                 "[RandomForestProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  constexpr int nclasses = 3;
  arma::mat trainX = arma::randu<arma::mat>(N, D);
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  // Initial model.
  RandomForestModel* model = new RandomForestModel;
  model->rf.Train(trainX, trainY, nclasses);
  // Set as input.
  SetInputParam("input_model", std::move(model));

  arma::mat testX = { 0.123, 0.456 };
  testX = arma::trans(testX);
  SetInputParam("test", std::move(testX));
  RUN_BINDING();

  arma::mat probabilities = params.Get<arma::mat>("probabilities");
  REQUIRE(probabilities.n_elem == nclasses);
}
