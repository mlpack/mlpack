/**
 * @file tests/main_tests/grad_boosting_tests.cpp
 * @author Abhimanyu Dayal
 *
 * Test RUN_BINDING() of grad_boosting_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/grad_boosting/grad_boosting_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(GradBoostingTestFixture);

/**
 * Run the binding on iris dataset, and check with test dataset. 
 * Confirm if the test accuracy is greater than 60%.
 */
TEST_CASE_METHOD(GradBoostingTestFixture, "GradBoostingIrisTest",
                 "[GradBoostingMainTest][BindingTests]")
{
  arma::mat db;
  if (!data::Load("iris_train.csv", db))
    FAIL("Cannot load test dataset iris_train.csv!");
  
  arma::Row<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris iris_train_labels.txt");

  arma::mat testDb;
  if (!data::Load("iris_test.csv", testDb))
    FAIL("Cannot load test dataset iris_test.csv!");

  arma::Row<size_t> testLabels;
  if (!data::Load("iris_test_labels.csv", testLabels))
    FAIL("Cannot load test dataset iris_test_labels.csv!");

  SetInputParam("training", std::move(db));
  SetInputParam("labels", std::move(labels));
  SetInputParam("numModels", 5);

  RUN_BINDING();

  arma::Row<size_t> predictions;
  predictions = std::move(params.Get<arma::Row<size_t>>("predictions"));

  GradBoostingModel* m = params.Get<GradBoostingModel*>("output_model");
  params.Get<GradBoostingModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input trained model.
  SetInputParam("test", std::move(testDb));
  SetInputParam("input_model", m);

  RUN_BINDING();

  arma::Row<size_t> predictions = params.Get<arma::Row<size_t>>("predictions");

  double accuracy = 0;
  for (size_t i = 0; i < testLabels.n_elem; i++) 
  {
    if(testLabels(i) == predictions(i)) 
    {
      accuracy++;
    }
  }

  accuracy = accuracy / ((double) testLabels.n_elem);
  accuracy *= 100.0;

  REQUIRE(accuracy > 60);

}