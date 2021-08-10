/**
 * @file tests/main_tests/adaboost_fit_test.cpp
 * @author Nippun Sharma
 *
 * Test RUN_BINDING() of adaboost_fit_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methoods/adaboost/adaboost_fit_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(AdaBoostFitMainTestFixture);

// Test if the error is thrown for invalid tolerance.
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitToleranceTest",
                 "[AdaBoostFitMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("tolerance", -1); // Invalid!

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROW_AS(RUN_BINDING(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Test if error is thrown for invalid iterations.
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitIterationsTest",
                 "[AdaBoostFitMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("iterations", -1); // Invalid!

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROW_AS(RUN_BINDING(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Test if error is thrown for invalid weak learner.
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitWeakLearnerTest",
                 "[AdaBoostFitMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weak_learner", "xyz"); // Invalid!

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROW_AS(RUN_BINDING(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Make sure that the output model is "trained" and usable.
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitModelReuseTest",
                 "[AdaBoostFitMainTest][BindingTests]")
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

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  AdaBoostModel* model = params.Get<AdaBoostModel*>("output_model");
  REQUIRE(model->Dimensionality() == trainData.n_rows);

  // Can we make predictions from this model.
  arma::Row<size_t> predictions, probabilities;
  model->Classify(testData, predictions, probabilities);

  // Check if dims of predictions is same as train data.
  REQUIRE(trainData.n_rows == probabilities.n_rows);
}

/**
 * Check that the last dimension of the training set is
 * used as labels when labels are not passed specifically 
 * and results are same from both label and without label models.
 */
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitMainWithoutLabelTest",
                 "[AdaBoostFitMainMainTest][BindingTests]")
{
  // Train adaboost without providing labels.
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    FAIL("Unable to load train dataset trainSet.csv!");

  // Give labels.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);

  arma::mat testData;
  if (!data::Load("testSet.csv", testData))
    FAIL("Unable to load test dataset testSet.csv!");

  // Delete the last row containing labels from test dataset.
  testData.shed_row(testData.n_rows - 1);

  SetInputParam("training", trainData);
  SetInputParam("iterations",(int) 20);
  SetInputParam("tolerance", 0.001);

  RUN_BINDING();

  // Get predictions from trained model.
  arma::Row<size_t> predictions;
  AdaBoostModel* model  = params.Get<AdaBoostModel*>("output_model");
  model->Classify(testData, predictions);

  // Train new model by providing labels seperately.
  AdaBoostModel* model2 = new AdaBoostModel(model->Mappings(),
                                            model->WeakLearnerType());
  arma::Row<size_t> predictions2;
  model2->Train(trainData, labels, model->Mappings().n_elem,
      (int) 20, 0.001);
  model2->Classify(testData, predictions2);

  // Check if both predictions are same or not.
  CheckMatrices(predictions, predictions2);
}
