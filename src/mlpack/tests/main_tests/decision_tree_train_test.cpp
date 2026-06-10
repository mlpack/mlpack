/**
 * @file tests/main_tests/decision_tree_train_test.cpp
 * @author Manish Kumar
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() for decision_tree_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(DecisionTreeTrainTestFixture);

/**
 * Check that output dimension is correct.
 */
TEST_CASE_METHOD(DecisionTreeTrainTestFixture,
                 "DecisionTreeTrainOutputDimensionTest",
                 "[DecisionTreeTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  DatasetInfo info(D);
  SetInputParam("training", std::make_tuple(info, std::move(trainX)));
  SetInputParam("labels", std::move(trainY));
  RUN_BINDING();

  arma::Row<size_t> preds;
  arma::mat::fixed<2, 1> testX = { 0.123, 0.456 };
  params.Get<DecisionTreeModel*>("output_model")->tree.Classify(testX, preds);
  REQUIRE(preds.n_elem == 1);
}

/**
 * Check that output dimension is correct when using categorical data.
 */
TEST_CASE_METHOD(DecisionTreeTrainTestFixture,
                 "DecisionTreeTrainCategoricalTest",
                 "[DecisionTreeTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  DatasetInfo info(D);
  // Make last row categorical with 5 categories.
  trainX.row(D - 1) = arma::randi<arma::rowvec>(N, arma::distr_param(0, 4));
  info.Type(D - 1) = Datatype::categorical;
  SetInputParam("training", std::make_tuple(info, std::move(trainX)));
  SetInputParam("labels", std::move(trainY));
  RUN_BINDING();

  arma::Row<size_t> preds;
  arma::mat::fixed<2, 1> testX = { 0.123, 0.456 };
  params.Get<DecisionTreeModel*>("output_model")->tree.Classify(testX, preds);
  REQUIRE(preds.n_elem == 1);
}

/**
 * Check that invalid parameter choices throw errors.
 */
TEST_CASE_METHOD(DecisionTreeTrainTestFixture,
                 "DecisionTreeTrainParameters",
                 "[DecisionTreeTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::trans(arma::randu<arma::mat>(N, D));
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  TextOptions opts;
  opts.Categorical() = true;
  SetInputParam("training",
      std::make_tuple(opts.DatasetInfo(), std::move(trainX)));
  SetInputParam("labels", std::move(trainY));

  SetInputParam("minimum_leaf_size", (int) -1);         // Invalid.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("minimum_leaf_size", (int) 3);          // Now valid.
  SetInputParam("minimum_gain_split", (double) -0.1);   // Invalid
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("minimum_gain_split", (double) 1.1);    // Invalid
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("minimum_gain_split", (double) 1e-7);   // Now valid.
  SetInputParam("maximum_depth", (int) -1);             // Invalid.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
