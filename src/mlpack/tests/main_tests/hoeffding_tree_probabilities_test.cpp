/**
 * @file tests/main_tests/hoeffding_tree_probabilities_test.cpp
 * @author Dirk Eddelbuettel
 *
 * Test RUN_BINDING() of hoeffding_tree_probabilities_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_probabilities_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(HoeffdingTreeProbabilitiesTestFixture);

/**
 * Check that absence of model throws error.
 */
TEST_CASE_METHOD(HoeffdingTreeProbabilitiesTestFixture,
                 "HoeffdingTreeProbabilitiesNoModelTest",
                 "[HoeffdingTreeProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat testX = arma::randu<arma::mat>(N, D);
  DatasetInfo info(D);
  SetInputParam("test", std::make_tuple(info, std::move(testX)));
  // (Required) model is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that lack of data throws error.
 */
TEST_CASE_METHOD(HoeffdingTreeProbabilitiesTestFixture,
                 "HoeffdingTreeProbabilitiesNoDataTest",
                 "[HoeffdingTreeProbabilitiesMainTest][BindingTests]")
{
  // Initial model.
  HoeffdingTreeModel* model = new HoeffdingTreeModel;
  // Set as input.
  SetInputParam("input_model", std::move(model));
  // No data, expect error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that prediction size corresponds to test data size.
 */
TEST_CASE_METHOD(HoeffdingTreeProbabilitiesTestFixture,
                 "HoeffdingTreeProbabilitiesPredictionSizeTest",
                 "[HoeffdingTreeProbabilitiesMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;
  arma::mat trainX = arma::randu<arma::mat>(N, D);
  arma::Row<size_t> trainY =
      arma::randi<arma::Row<size_t>>(N, arma::distr_param(0, 4));
  DatasetInfo info(D);
  // Initial model.
  HoeffdingTreeModel* model = new HoeffdingTreeModel;
  model->BuildModel(trainX,
                    info,
                    trainY,
                    5,
                    true,  // batchTraining
                    0.95,  // successProbability
                    0,     // maxSamples
                    100,   // checkInterval
                    100,   // minSamples
                    10,    // bins
                    100);  // observationsBeforeBinning
  // Set as input.
  SetInputParam("input_model", std::move(model));

  arma::mat testX = { 0.123, 0.456 };
  testX = arma::trans(testX);
  SetInputParam("test", std::make_tuple(info, std::move(testX)));
  //testX = arma::trans(testX);
  RUN_BINDING();

  arma::mat probs = params.Get<arma::mat>("probabilities");
  REQUIRE(probs.n_elem == 1);
}
