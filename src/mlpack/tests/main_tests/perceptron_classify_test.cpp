/**
 * @file tests/main_tests/perceptron_classify_test.cpp
 * @author Manish Kumar
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of perceptron_classify_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron_classify_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(PerceptronClassifyTestFixture);

/**
 * Check that absence of model throws error.
 */
TEST_CASE_METHOD(PerceptronClassifyTestFixture,
                 "PerceptronClassifyNoModelTest",
                 "[PerceptronClassifyMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat testX = arma::randu<arma::mat>(N, D);
  SetInputParam("test", std::move(testX));
  // (Required) model is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that lack of data throws error.
 */
TEST_CASE_METHOD(PerceptronClassifyTestFixture,
                 "PerceptronClassifyNoDataTest",
                 "[PerceptronClassifyMainTest][BindingTests]")
{
  // Initial model.
  PerceptronModel* model = new PerceptronModel;
  // Set as input.
  SetInputParam("input_model", std::move(model));
  // No data, expect error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that prediction size corresponds to test data size.
 */
TEST_CASE_METHOD(PerceptronClassifyTestFixture,
                 "PerceptronClassifyPredictionSizeTest",
                 "[PerceptronClassifyMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY =
      arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));

  // Initial model.
  PerceptronModel* model = new PerceptronModel;
  model->P().Train(trainX, trainY, 5);
  NormalizeLabels(trainY, trainY, model->Map());

  // Set as input.
  SetInputParam("input_model", std::move(model));

  arma::mat testX = { 0.123, 0.456 };
  testX = arma::trans(testX);
  SetInputParam("test", std::move(testX));
  RUN_BINDING();

  arma::Row<size_t> predictions = params.Get<arma::Row<size_t>>("predictions");
  REQUIRE(predictions.n_elem == 1);
}
