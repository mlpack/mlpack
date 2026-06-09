/**
 * @file tests/main_tests/decision_tree_classify_test.cpp
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() for decision_tree_classify_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree_classify_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(DecisionTreeClassifyTestFixture);

/**
 * Check that lack of model throws error.
 */
TEST_CASE_METHOD(DecisionTreeClassifyTestFixture,
                 "DecisionTreeClassifyNoModelTest",
                 "[DecisionTreeClassifyMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat test = arma::trans(arma::randu<arma::mat>(N, D));
  TextOptions opts;
  opts.Categorical() = true;
  SetInputParam("test", std::make_tuple(opts.DatasetInfo(), std::move(test)));
  // (Required) model is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that lack of data throws error.
 */
TEST_CASE_METHOD(DecisionTreeClassifyTestFixture,
                 "DecisionTreeClassifyNoDataTest",
                 "[DecisionTreeClassifyMainTest][BindingTests]")
{
  // Initial model.
  DecisionTreeModel* model = new DecisionTreeModel;
  // Set as input.
  SetInputParam("input_model", std::move(model));
  // No data, expect error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that prediction size corresponds to test data size.
 */
TEST_CASE_METHOD(DecisionTreeClassifyTestFixture,
                 "DecisionTreeClassifyPredictionSizeTest",
                 "[DecisionTreeClassifyMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  // Initial model.
  DecisionTreeModel* model = new DecisionTreeModel;
  model->tree.Train(trainX, trainY, 3);
  // Set as input.
  SetInputParam("input_model", std::move(model));

  arma::mat testX = { 0.123, 0.456 };
  testX = arma::trans(testX);
  TextOptions opts;
  opts.Categorical() = true;
  SetInputParam("test", std::make_tuple(opts.DatasetInfo(), std::move(testX)));
  RUN_BINDING();

  arma::Row<size_t> predictions = params.Get<arma::Row<size_t>>("predictions");
  REQUIRE(predictions.n_elem == 1);
}
