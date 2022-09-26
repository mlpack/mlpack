/**
 * @file tests/main_tests/adaboost_probabilities_test.cpp
 * @author Nippun Sharma
 *
 * Test RUN_BINDING() of adaboost_probabilities_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/adaboost/adaboost_probabilities_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(AdaBoostPredictProbaTestFixture);

/**
 * Check that total number of rows of probabilities matrix is equal to total
 * number of rows of input data and that each column of probabilities matrix sums
 * up to 1.
 */
TEST_CASE_METHOD(AdaBoostPredictProbaTestFixture,
                 "AdaBoostPredictProbaProbabilitiesTest",
                 "[AdaBoostPredictProbaMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Unable to load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  arma::Col<size_t> mappings = {0, 1, 2};
  AdaBoostModel* model = new AdaBoostModel(mappings, 0);
  model->Train(trainData, labels, 3, (int) 20, (double) 0.0001);

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  RUN_BINDING();

  arma::mat probabilities;
  probabilities = std::move(params.Get<arma::mat>("probabilities"));

  REQUIRE(probabilities.n_cols == testSize);

  for (size_t i = 0; i < testSize; ++i)
    REQUIRE(arma::accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
}

/**
 * Check that an exception is thrown when the number of features of
 * input data is not equal to the number of features of the test data.
 */
TEST_CASE_METHOD(AdaBoostPredictProbaTestFixture,
                 "AdaBoostPredictProbaShapeTest",
                 "[AdaBoostPredictProbaMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Unable to load test dataset vc2.csv!");

  // reducing the number of features in test data.
  testData.shed_row(testData.n_rows - 1);

  arma::Col<size_t> mappings = {0, 1, 2};
  AdaBoostModel* model = new AdaBoostModel(mappings, 0);
  model->Train(trainData, labels, 3, (int) 20, (double) 0.0001);

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testData));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
