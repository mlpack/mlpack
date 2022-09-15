/**
 * @file tests/main_tests/adaboost_train_test.cpp
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
#include <mlpack/methods/adaboost/adaboost_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(AdaBoostFitTestFixture);

// Test if the error is thrown for invalid tolerance.
TEST_CASE_METHOD(AdaBoostFitTestFixture, "AdaBoostFitToleranceTest",
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
  SetInputParam("tolerance", (double) -0.001); // Invalid!

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Test if error is thrown for invalid iterations.
TEST_CASE_METHOD(AdaBoostFitTestFixture, "AdaBoostFitIterationsTest",
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Test if error is thrown for invalid weak learner.
TEST_CASE_METHOD(AdaBoostFitTestFixture, "AdaBoostFitWeakLearnerTest",
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
  SetInputParam("weak_learner", std::string("xyz")); // Invalid!

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Make sure that the output model is "trained" and usable.
TEST_CASE_METHOD(AdaBoostFitTestFixture, "AdaBoostFitModelReuseTest",
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

  size_t trainDims = trainData.n_rows;
  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  AdaBoostModel* model = params.Get<AdaBoostModel*>("output_model");
  REQUIRE(model->Dimensionality() == trainDims);

  // Can we make predictions from this model.
  arma::Row<size_t> predictions;
  arma::mat probabilities;
  model->Classify(testData, predictions, probabilities);

  // Check if dims of predictions is same as train data.
  REQUIRE(model->Mappings().n_elem == probabilities.n_rows);
}

/**
 * Different Weak learner should give different outputs.
 */
TEST_CASE_METHOD(AdaBoostFitTestFixture, "AdaBoostFitDiffWeakLearnerOutputTest",
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

  SetInputParam("training", trainData);
  SetInputParam("labels", labels);

  RUN_BINDING();

  arma::Row<size_t> output;
  params.Get<AdaBoostModel*>("output_model")->Classify(testData, output);

  CleanMemory();
  ResetSettings();

  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("weak_learner", std::string("perceptron"));

  RUN_BINDING();

  arma::Row<size_t> outputPerceptron;
  params.Get<AdaBoostModel*>("output_model")->Classify(testData,
      outputPerceptron);

  REQUIRE(arma::accu(output != outputPerceptron) > 1);
}

/**
 * Accuracy increases as Number of Iterations increases.
 * (Or converges and remains same)
 */
TEST_CASE_METHOD(AdaBoostFitTestFixture, "AdaBoostFitDiffItrTest",
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

  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    FAIL("Unable to load labels for vc2__test_labels.txt");

  // Iterations = 1
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("weak_learner", std::string("perceptron"));
  SetInputParam("iterations", (int) 1);

  RUN_BINDING();

  // Calculate accuracy.
  arma::Row<size_t> output;
  params.Get<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  size_t correct = arma::accu(output == testLabels);
  double accuracy1 = (double(correct) / double(testLabels.n_elem) * 100);

  CleanMemory();
  ResetSettings();

  // Iterations = 10
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("weak_learner", std::string("perceptron"));
  SetInputParam("iterations", (int) 10);

  RUN_BINDING();

  // Calculate accuracy.
  params.Get<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy10 = (double(correct) / double(testLabels.n_elem) * 100);

  CleanMemory();
  ResetSettings();

  // Iterations = 100
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("weak_learner", std::string("perceptron"));
  SetInputParam("iterations", (int) 100);

  RUN_BINDING();

  // Calculate accuracy.
  params.Get<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy100 = (double(correct) / double(testLabels.n_elem) * 100);

  REQUIRE(accuracy1 <= accuracy10);
  REQUIRE(accuracy10 <= accuracy100);
}

/**
 * Accuracy increases as tolerance decreases.
 * (Execution Time also increases)
 */
TEST_CASE_METHOD(AdaBoostFitTestFixture, "AdaBoostFitDiffTolTest",
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

  arma::Row<size_t> testLabels;
  if (!data::Load("vc2_test_labels.txt", testLabels))
    FAIL("Unable to load labels for vc2__test_labels.txt");

  // tolerance = 0.001
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("tolerance", (double) 0.001);

  RUN_BINDING();

  // Calculate accuracy.
  arma::Row<size_t> output;
  params.Get<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  size_t correct = arma::accu(output == testLabels);
  double accuracy1 = (double(correct) / double(testLabels.n_elem) * 100);

  CleanMemory();
  ResetSettings();

  // tolerance = 0.01
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("tolerance", (double) 0.01);

  RUN_BINDING();

  // Calculate accuracy.
  params.Get<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy2 = (double(correct) / double(testLabels.n_elem) * 100);

  CleanMemory();
  ResetSettings();

  // tolerance = 0.1
  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("tolerance", (double) 0.1);

  RUN_BINDING();

  // Calculate accuracy.
  params.Get<AdaBoostModel*>("output_model")->Classify(testData,
       output);

  correct = arma::accu(output == testLabels);
  double accuracy3 = (double(correct) / double(testLabels.n_elem) * 100);

  REQUIRE(accuracy1 <= accuracy2);
  REQUIRE(accuracy2 <= accuracy3);
}
