/**
 * @file tests/main_tests/adaboost_test.cpp
 * @author Nikhil Goel
 *
 * Test RUN_BINDING() of adaboost_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/adaboost/adaboost_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(AdaBoostTestFixture);

/**
 * Check that number of output labels and number of input
 * points are equal.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostOutputDimensionTest",
                 "[AdaBoostMainTest][BindingTests]")
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

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  SetInputParam("test", std::move(testData));

  RUN_BINDING();

  // Check that number of predicted labels is equal to the input test points.
  REQUIRE(params.Get<arma::Row<size_t>>("output").n_cols == testSize);
  REQUIRE(params.Get<arma::Row<size_t>>("output").n_rows == 1);
}

/**
 * Check that total number of rows of probabilities matrix is equal to total
 * number of rows of input data and that each column of probabilities matrix sums
 * up to 1.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostProbabilitiesTest",
                 "[AdaBoostMainTest][BindingTests]")
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

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  SetInputParam("test", std::move(testData));

  RUN_BINDING();

  arma::mat probabilities;
  probabilities = std::move(params.Get<arma::mat>("probabilities"));

  REQUIRE(probabilities.n_cols == testSize);

  for (size_t i = 0; i < testSize; ++i)
    REQUIRE(arma::accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostModelReuseTest",
                 "[AdaBoostMainTest][BindingTests]")
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

  SetInputParam("test", testData);

  RUN_BINDING();

  arma::Row<size_t> output;
  output = std::move(params.Get<arma::Row<size_t>>("output"));

  AdaBoostModel* model = params.Get<AdaBoostModel*>("output_model");
  ResetSettings();

  SetInputParam("test", std::move(testData));
  SetInputParam("input_model", model);

  RUN_BINDING();

  // Check that initial output and output using saved model are same.
  CheckMatrices(output, params.Get<arma::Row<size_t>>("output"));
}

/**
 * Test that iterations in adaboost is always non-negative.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostItrTest",
                 "[AdaBoostMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    FAIL("Unable load train dataset trainSet.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("iterations", (int) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that the last dimension of the training set is
 * used as labels when labels are not passed specifically 
 * and results are same from both label and without label models.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostWithoutLabelTest",
                 "[AdaBoostMainTest][BindingTests]")
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

  SetInputParam("test", testData);

  RUN_BINDING();

  arma::Row<size_t> output;
  output = std::move(params.Get<arma::Row<size_t>>("output"));

  CleanMemory();
  ResetSettings();

  trainData.shed_row(trainData.n_rows - 1);

  // Now train AdaBoost with labels provided.
  SetInputParam("training", std::move(trainData));
  SetInputParam("test", std::move(testData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  // Check that initial output and final output matrix are same.
  CheckMatrices(output, params.Get<arma::Row<size_t>>("output"));
}

/**
 * Testing that only one of training data or pre-trained model is passed.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostTrainingDataOrModelTest",
                 "[AdaBoostMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    FAIL("Unable to load train dataset trainSet.csv!");

  SetInputParam("training", std::move(trainData));

  RUN_BINDING();

  SetInputParam("input_model",
                params.Get<AdaBoostModel*>("output_model"));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * This test can be removed in mlpack 4.0.0.  This tests that the output and
 * predictions outputs are the same.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostOutputPredictionsTest",
                 "[AdaBoostMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));

  RUN_BINDING();

  CheckMatrices(params.Get<arma::Row<size_t>>("output"),
                params.Get<arma::Row<size_t>>("predictions"));
}

/**
 * Weak learner should be either Decision Stump or Perceptron.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostWeakLearnerTest",
                 "[AdaBoostMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("trainSet.csv", trainData))
    FAIL("Unable to load train dataset trainSet.csv!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("weak_learner", std::string("decision tree"));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Different Weak learner should give different outputs.
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostDiffWeakLearnerOutputTest",
                 "[AdaBoostMainTest][BindingTests]")
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
  SetInputParam("test", testData);

  RUN_BINDING();

  arma::Row<size_t> output;
  output = std::move(params.Get<arma::Row<size_t>>("output"));

  CleanMemory();
  ResetSettings();

  SetInputParam("training", trainData);
  SetInputParam("labels", labels);
  SetInputParam("test", testData);
  SetInputParam("weak_learner", std::string("perceptron"));

  RUN_BINDING();

  arma::Row<size_t> outputPerceptron;
  outputPerceptron = std::move(params.Get<arma::Row<size_t>>("output"));

  REQUIRE(arma::accu(output != outputPerceptron) > 1);
}

/**
 * Accuracy increases as Number of Iterations increases.
 * (Or converges and remains same)
 */
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostDiffItrTest",
                 "[AdaBoostMainTest][BindingTests]")
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
TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostDiffTolTest",
                 "[AdaBoostMainTest][BindingTests]")
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
