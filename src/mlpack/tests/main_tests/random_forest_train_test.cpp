/**
 * @file tests/main_tests/random_forest_train_test.cpp
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of random_forest_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(RandomForestTrainTestFixture);

/**
 * Check output dimension.
 */
TEST_CASE_METHOD(RandomForestTrainTestFixture,
                 "RandomForestTrainOutputDimensionTest",
                 "[RandomForestTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::randu<arma::mat>(N, D);
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  RUN_BINDING();

  arma::Row<size_t> preds;
  arma::mat::fixed<2, 1> testX = { 0.123, 0.456 };
  params.Get<RandomForestModel*>("output_model")->rf.Classify(testX, preds);
  REQUIRE(preds.n_elem == 1);
}

/**
 * Test num_trees, min_leaf_size, max_leaf_size checks
 */
TEST_CASE_METHOD(RandomForestTrainTestFixture,
                 "RandomForestTrainParameterTest",
                 "[RandomForestTrainMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::randu<arma::mat>(N, D);
  arma::Row<size_t> trainY = arma::randu<arma::Row<size_t>>(N);
  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("num_trees", (int) 0); // Invalid.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("num_trees", (int) 5);         // Now valid.
  SetInputParam("minimum_leaf_size", (int) 0); // Invalid.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("minimum_leaf_size", (int) 3); // Now valid.
  SetInputParam("maximum_depth", (int) -1);    // Invalid.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("maximum_depth", (int) 0);             // Now valid.
  SetInputParam("minimum_gain_split", (double) -0.1);  // Invalid.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("minimum_gain_split", (double) 0);     // Now valid
  SetInputParam("subspace_dim", (int) -1);             // Invalid.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
