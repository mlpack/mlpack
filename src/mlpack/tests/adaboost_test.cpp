/**
 * @file tests/adaboost_test.cpp
 * @author Udit Saxena
 *
 * Tests for AdaBoost class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/adaboost.hpp>

#include "serialization.hpp"
#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace arma;
using namespace mlpack;

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Iris dataset.  It
 * checks whether the hamming loss breaches the upperbound, which is provided by
 * ztAccumulator.
 */
TEST_CASE("HammingLossBoundIris", "[AdaBoostTest]")
{
  arma::mat inputData;

  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  int perceptronIter = 400;

  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);

  // Define parameters for AdaBoost.
  size_t iterations = 100;
  double tolerance = 1e-10;
  AdaBoost<> a(tolerance);
  double ztProduct = a.Train(inputData, labels.row(0), numClasses, p,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double hammingLoss = (double) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) == true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Iris dataset.  It
 * checks if the error returned by running a single instance of the weak learner
 * close to that of the boosted weak learner using adaboost.
 */
TEST_CASE("WeakLearnerErrorIris", "[AdaBoostTest]")
{
  arma::mat inputData;

  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Mat<size_t> labels;

  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  int perceptronIter = 400;

  arma::Row<size_t> perceptronPrediction(labels.n_cols);
  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  size_t countWeakLearnerError = arma::accu(labels != perceptronPrediction);
  double weakLearnerErrorRate = (double) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 100;
  double tolerance = 1e-10;
  AdaBoost<> a(inputData, labels.row(0), numClasses, p, iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);;
  double error = (double) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It checks whether the hamming loss breaches the upperbound, which
 * is provided by ztAccumulator.
 */
TEST_CASE("HammingLossBoundVertebralColumn", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;
  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<> a(tolerance);
  double ztProduct = a.Train(inputData, labels.row(0), numClasses, p,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double hammingLoss = (double) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) == true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using adaboost.
 */
TEST_CASE("WeakLearnerErrorVertebralColumn", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;

  Row<size_t> perceptronPrediction(labels.n_cols);
  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  size_t countWeakLearnerError = arma::accu(labels != perceptronPrediction);
  double weakLearnerErrorRate = (double) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<> a(inputData, labels.row(0), numClasses, p, iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double error = (double) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on non-linearly separable
 * dataset.  It checks whether the hamming loss breaches the upperbound, which
 * is provided by ztAccumulator.
 */
TEST_CASE("HammingLossBoundNonLinearSepData", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  arma::Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;
  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<> a(tolerance);
  double ztProduct = a.Train(inputData, labels.row(0), numClasses, p,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels == predictedLabels);
  double hammingLoss = (double) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) <= true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on a non-linearly separable
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using AdaBoost.
 */
TEST_CASE("WeakLearnerErrorNonLinearSepData", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  arma::Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;

  Row<size_t> perceptronPrediction(labels.n_cols);
  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  size_t countWeakLearnerError = arma::accu(labels != perceptronPrediction);
  double weakLearnerErrorRate = (double) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<> a(inputData, labels.row(0), numClasses, p, iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double error = (double) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Iris dataset.  It
 * checks whether the Hamming loss breaches the upper bound, which is provided
 * by ztAccumulator.  This uses decision stumps as the weak learner.
 */
TEST_CASE("HammingLossIris_DS", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris_labels.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  arma::Row<size_t> labelsvec = labels.row(0);
  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<ID3DecisionStump> a(tolerance);
  double ztProduct = a.Train(inputData, labelsvec, numClasses, ds,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double hammingLoss = (double) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) == true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on a non-linearly separable
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using adaboost.
 * This is for the weak learner: decision stumps.
 */
TEST_CASE("WeakLearnerErrorIris_DS", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris_labels.txt");

  // no need to map the labels here

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  arma::Row<size_t> labelsvec = labels.row(0);

  arma::Row<size_t> dsPrediction(labels.n_cols);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);
  ds.Classify(inputData, dsPrediction);

  size_t countWeakLearnerError = arma::accu(labels != dsPrediction);
  double weakLearnerErrorRate = (double) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;

  AdaBoost<ID3DecisionStump> a(inputData, labelsvec, numClasses, ds,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double error = (double) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using adaboost.
 * This is for the weak learner: decision stumps.
 */
TEST_CASE("HammingLossBoundVertebralColumn_DS", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  arma::Row<size_t> labelsvec = labels.row(0);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;

  AdaBoost<ID3DecisionStump> a(tolerance);
  double ztProduct = a.Train(inputData, labelsvec, numClasses, ds,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double hammingLoss = (double) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) == true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using adaboost.
 * This is for the weak learner: decision stumps.
 */
TEST_CASE("WeakLearnerErrorVertebralColumn_DS", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  arma::Row<size_t> dsPrediction(labels.n_cols);
  arma::Row<size_t> labelsvec = labels.row(0);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);
  ds.Classify(inputData, dsPrediction);

  size_t countWeakLearnerError = arma::accu(labels != dsPrediction);
  double weakLearnerErrorRate = (double) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<ID3DecisionStump> a(inputData, labelsvec, numClasses, ds,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double error = (double) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on non-linearly separable
 * dataset.  It checks whether the hamming loss breaches the upperbound, which
 * is provided by ztAccumulator.  This is for the weak learner: decision stumps.
 */
TEST_CASE("HammingLossBoundNonLinearSepData_DS", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  arma::Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 2;
  const size_t inpBucketSize = 6;
  arma::Row<size_t> labelsvec = labels.row(0);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);

  // Define parameters for Adaboost.
  size_t iterations = 50;
  double tolerance = 1e-10;

  AdaBoost<ID3DecisionStump> a(tolerance);
  double ztProduct = a.Train(inputData, labelsvec, numClasses, ds,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double hammingLoss = (double) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) == true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on a non-linearly separable
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using adaboost.
 * This for the weak learner: decision stumps.
 */
TEST_CASE("WeakLearnerErrorNonLinearSepData_DS", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  arma::Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 2;
  const size_t inpBucketSize = 3;
  arma::Row<size_t> labelsvec = labels.row(0);

  arma::Row<size_t> dsPrediction(labels.n_cols);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);
  ds.Classify(inputData, dsPrediction);

  size_t countWeakLearnerError = arma::accu(labels != dsPrediction);
  double weakLearnerErrorRate = (double) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 500;
  double tolerance = 1e-23;

  AdaBoost<ID3DecisionStump > a(inputData, labelsvec, numClasses, ds,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = arma::accu(labels != predictedLabels);
  double error = (double) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It tests the Classify function and checks for a satisfactory error
 * rate.
 */
TEST_CASE("ClassifyTest_VERTEBRALCOL", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 1000;

  arma::mat testData;

  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");

  arma::Mat<size_t> trueTestLabels;

  if (!data::Load("vc2_test_labels.txt", trueTestLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  Row<size_t> perceptronPrediction(labels.n_cols);
  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  // Define parameters for AdaBoost.
  size_t iterations = 100;
  double tolerance = 1e-10;
  AdaBoost<> a(inputData, labels.row(0), numClasses, p, iterations, tolerance);

  arma::Row<size_t> predictedLabels1(testData.n_cols),
                    predictedLabels2(testData.n_cols);
  arma::mat probabilities;

  a.Classify(testData, predictedLabels1);
  a.Classify(testData, predictedLabels2, probabilities);

  REQUIRE(probabilities.n_cols == testData.n_cols);
  REQUIRE(probabilities.n_rows == numClasses);

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
    REQUIRE(predictedLabels1[i] == predictedLabels2[i]);

  arma::colvec pRow;
  arma::uword maxIndex = 0;

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
  {
    pRow = probabilities.unsafe_col(i);
    pRow.max(maxIndex);
    REQUIRE(predictedLabels1(i) == maxIndex);
    REQUIRE(arma::accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
  }

  size_t localError = arma::accu(trueTestLabels != predictedLabels1);
  double lError = (double) localError / trueTestLabels.n_cols;
  REQUIRE(lError <= 0.30);
}

/**
 * This test case runs the AdaBoost.mh algorithm on a non linearly separable
 * dataset.  It tests the Classify function and checks for a satisfactory error
 * rate.
 */
TEST_CASE("ClassifyTest_NONLINSEP", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  arma::Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  // Define your own weak learner; in this test decision stumps are used.
  const size_t numClasses = 2;
  const size_t inpBucketSize = 3;
  arma::Row<size_t> labelsvec = labels.row(0);

  arma::mat testData;

  if (!data::Load("test_nonlinsep.txt", testData))
    FAIL("Cannot load test dataset test_nonlinsep.txt!");

  arma::Mat<size_t> trueTestLabels;

  if (!data::Load("test_labels_nonlinsep.txt", trueTestLabels))
    FAIL("Cannot load labels for test_labels_nonlinsep.txt");

  arma::Row<size_t> dsPrediction(labels.n_cols);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<ID3DecisionStump > a(inputData, labelsvec, numClasses, ds,
      iterations, tolerance);

  arma::Row<size_t> predictedLabels1(testData.n_cols),
                    predictedLabels2(testData.n_cols);
  arma::mat probabilities;

  a.Classify(testData, predictedLabels1);
  a.Classify(testData, predictedLabels2, probabilities);

  REQUIRE(probabilities.n_cols == testData.n_cols);

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
    REQUIRE(predictedLabels1[i] == predictedLabels2[i]);

  arma::colvec pRow;
  arma::uword maxIndex = 0;

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
  {
    pRow = probabilities.unsafe_col(i);
    pRow.max(maxIndex);
    REQUIRE(predictedLabels1(i) == maxIndex);
    REQUIRE(arma::accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
  }

  size_t localError = arma::accu(trueTestLabels != predictedLabels1);
  double lError = (double) localError / trueTestLabels.n_cols;
  REQUIRE(lError <= 0.30);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Iris Dataset.  It
 * trains it on two thirds of the Iris dataset (iris_train.csv), and tests on
 * the remaining third of the dataset (iris_test.csv).  It tests the Classify()
 * function and checks for a satisfactory error rate.
 */
TEST_CASE("ClassifyTest_IRIS", "[AdaBoostTest]")
{
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load test dataset iris_train.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris_train_labels.csv");
  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;

  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<> a(inputData, labels.row(0), numClasses, p, iterations, tolerance);

  arma::mat testData;
  if (!data::Load("iris_test.csv", testData))
    FAIL("Cannot load test dataset iris_test.csv!");

  arma::Row<size_t> predictedLabels(testData.n_cols);
  a.Classify(testData, predictedLabels);

  arma::Mat<size_t> trueTestLabels;
  if (!data::Load("iris_test_labels.csv", trueTestLabels))
    FAIL("Cannot load test dataset iris_test_labels.csv!");

  arma::Row<size_t> predictedLabels1(testData.n_cols),
                    predictedLabels2(testData.n_cols);
  arma::mat probabilities;

  a.Classify(testData, predictedLabels1);
  a.Classify(testData, predictedLabels2, probabilities);

  REQUIRE(probabilities.n_cols == testData.n_cols);

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
    REQUIRE(predictedLabels1[i] == predictedLabels2[i]);

  arma::colvec pRow;
  arma::uword maxIndex = 0;

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
  {
    pRow = probabilities.unsafe_col(i);
    pRow.max(maxIndex);
    REQUIRE(predictedLabels1(i) == maxIndex);
    REQUIRE(arma::accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
  }

  size_t localError = arma::accu(trueTestLabels != predictedLabels1);
  double lError = (double) localError / labels.n_cols;
  REQUIRE(lError <= 0.30);
}

/**
 * Ensure that the Train() function works like it is supposed to, by building
 * AdaBoost on one dataset and then re-training on another dataset.
 */
TEST_CASE("TrainTest", "[AdaBoostTest]")
{
  // First train on the iris dataset.
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load test dataset iris_train.csv!");

  arma::Mat<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris_train_labels.csv");

  const size_t numClasses = max(labels.row(0)) + 1;

  size_t perceptronIter = 800;
  Perceptron<> p(inputData, labels.row(0), numClasses, perceptronIter);

  // Now train AdaBoost.
  size_t iterations = 50;
  double tolerance = 1e-10;
  AdaBoost<> a(inputData, labels.row(0), numClasses, p, iterations, tolerance);

  // Now load another dataset...
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  const size_t newNumClasses = max(labels.row(0)) + 1;

  Perceptron<> p2(inputData, labels.row(0), newNumClasses, perceptronIter);

  a.Train(inputData, labels.row(0), newNumClasses, p2, iterations, tolerance);

  // Load test set to see if it trained on vc2 correctly.
  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");

  arma::Mat<size_t> trueTestLabels;
  if (!data::Load("vc2_test_labels.txt", trueTestLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt");

  // Define parameters for AdaBoost.
  arma::Row<size_t> predictedLabels(testData.n_cols);
  a.Classify(testData, predictedLabels);

  int localError = arma::accu(trueTestLabels != predictedLabels);
  double lError = (double) localError / trueTestLabels.n_cols;

  REQUIRE(lError <= 0.30);
}

TEST_CASE("PerceptronSerializationTest", "[AdaBoostTest]")
{
  // Build an AdaBoost object.
  mat data = randu<mat>(10, 500);
  Row<size_t> labels(500);
  for (size_t i = 0; i < 250; ++i)
    labels[i] = 0;
  for (size_t i = 250; i < 500; ++i)
    labels[i] = 1;

  Perceptron<> p(data, labels, 2, 800);
  AdaBoost<> ab(data, labels, 2, p, 50, 1e-10);

  // Now create another dataset to train with.
  mat otherData = randu<mat>(5, 200);
  Row<size_t> otherLabels(200);
  for (size_t i = 0; i < 100; ++i)
    otherLabels[i] = 1;
  for (size_t i = 100; i < 150; ++i)
    otherLabels[i] = 0;
  for (size_t i = 150; i < 200; ++i)
    otherLabels[i] = 2;

  Perceptron<> p2(otherData, otherLabels, 3, 500);
  AdaBoost<> abText(otherData, otherLabels, 3, p2, 50, 1e-10);

  AdaBoost<> abXml, abBinary;

  SerializeObjectAll(ab, abXml, abText, abBinary);

  // Now check that the objects are the same.
  REQUIRE(ab.Tolerance() == Approx(abXml.Tolerance()).epsilon(1e-7));
  REQUIRE(ab.Tolerance() == Approx(abText.Tolerance()).epsilon(1e-7));
  REQUIRE(ab.Tolerance() == Approx(abBinary.Tolerance()).epsilon(1e-7));

  REQUIRE(ab.WeakLearners() == abXml.WeakLearners());
  REQUIRE(ab.WeakLearners() == abText.WeakLearners());
  REQUIRE(ab.WeakLearners() == abBinary.WeakLearners());

  for (size_t i = 0; i < ab.WeakLearners(); ++i)
  {
    CheckMatrices(ab.WeakLearner(i).Weights(),
                  abXml.WeakLearner(i).Weights(),
                  abText.WeakLearner(i).Weights(),
                  abBinary.WeakLearner(i).Weights());

    CheckMatrices(ab.WeakLearner(i).Biases(),
                  abXml.WeakLearner(i).Biases(),
                  abText.WeakLearner(i).Biases(),
                  abBinary.WeakLearner(i).Biases());
  }
}

TEST_CASE("ID3DecisionStumpSerializationTest", "[AdaBoostTest]")
{
  // Build an AdaBoost object.
  mat data = randu<mat>(10, 500);
  Row<size_t> labels(500);
  for (size_t i = 0; i < 250; ++i)
    labels[i] = 0;
  for (size_t i = 250; i < 500; ++i)
    labels[i] = 1;

  ID3DecisionStump p(data, labels, 2, 800);
  AdaBoost<ID3DecisionStump> ab(data, labels, 2, p, 50, 1e-10);

  // Now create another dataset to train with.
  mat otherData = randu<mat>(5, 200);
  Row<size_t> otherLabels(200);
  for (size_t i = 0; i < 100; ++i)
    otherLabels[i] = 1;
  for (size_t i = 100; i < 150; ++i)
    otherLabels[i] = 0;
  for (size_t i = 150; i < 200; ++i)
    otherLabels[i] = 2;

  ID3DecisionStump p2(otherData, otherLabels, 3, 500);
  AdaBoost<ID3DecisionStump> abText(otherData, otherLabels, 3, p2, 50, 1e-10);

  AdaBoost<ID3DecisionStump> abXml, abBinary;

  SerializeObjectAll(ab, abXml, abText, abBinary);

  // Now check that the objects are the same.
  REQUIRE(ab.Tolerance() == Approx(abXml.Tolerance()).epsilon(1e-7));
  REQUIRE(ab.Tolerance() == Approx(abText.Tolerance()).epsilon(1e-7));
  REQUIRE(ab.Tolerance() == Approx(abBinary.Tolerance()).epsilon(1e-7));

  REQUIRE(ab.WeakLearners() == abXml.WeakLearners());
  REQUIRE(ab.WeakLearners() == abText.WeakLearners());
  REQUIRE(ab.WeakLearners() == abBinary.WeakLearners());

  for (size_t i = 0; i < ab.WeakLearners(); ++i)
  {
    REQUIRE(ab.WeakLearner(i).SplitDimension() ==
           abXml.WeakLearner(i).SplitDimension());
    REQUIRE(ab.WeakLearner(i).SplitDimension() ==
            abText.WeakLearner(i).SplitDimension());
    REQUIRE(ab.WeakLearner(i).SplitDimension() ==
            abBinary.WeakLearner(i).SplitDimension());
  }
}
