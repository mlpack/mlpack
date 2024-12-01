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
TEMPLATE_TEST_CASE("HammingLossBoundIris", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;

  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  int perceptronIter = 400;

  // Define parameters for AdaBoost.
  size_t iterations = 100;
  eT tolerance = 2e-10;
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  AdaBoost<PerceptronType, MatType> a;
  eT ztProduct = a.Train(inputData, labels.row(0), numClasses, iterations,
      tolerance, perceptronIter);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT hammingLoss = (eT) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) == true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Iris dataset.  It
 * checks if the error returned by running a single instance of the weak learner
 * close to that of the boosted weak learner using adaboost.
 */
TEMPLATE_TEST_CASE("WeakLearnerErrorIris", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  int perceptronIter = 400;

  Row<size_t> perceptronPrediction(labels.n_cols);
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  size_t countWeakLearnerError = accu(labels != perceptronPrediction);
  eT weakLearnerErrorRate = (eT) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 100;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(inputData, labels.row(0), numClasses,
      iterations, tolerance, perceptronIter);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT error = (eT) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It checks whether the hamming loss breaches the upperbound, which
 * is provided by ztAccumulator.
 */
TEMPLATE_TEST_CASE("HammingLossBoundVertebralColumn", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(tolerance);
  eT ztProduct = a.Train(inputData, labels.row(0), numClasses, iterations,
      tolerance, perceptronIter);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT hammingLoss = (eT) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) == true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using adaboost.
 */
TEMPLATE_TEST_CASE("WeakLearnerErrorVertebralColumn", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;

  Row<size_t> perceptronPrediction(labels.n_cols);
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  size_t countWeakLearnerError = accu(labels != perceptronPrediction);
  eT weakLearnerErrorRate = (eT) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(inputData, labels.row(0), numClasses,
      iterations, tolerance, perceptronIter);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT error = (eT) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on non-linearly separable
 * dataset.  It checks whether the hamming loss breaches the upperbound, which
 * is provided by ztAccumulator.
 */
TEMPLATE_TEST_CASE("HammingLossBoundNonLinearSepData", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(tolerance);
  eT ztProduct = a.Train(inputData, labels.row(0), numClasses, iterations,
      tolerance, perceptronIter);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels == predictedLabels);
  eT hammingLoss = (eT) countError / labels.n_cols;

  // Check that ztProduct is finite.
  REQUIRE(std::isfinite(ztProduct) <= true);
  REQUIRE(hammingLoss <= ztProduct);
}

/**
 * This test case runs the AdaBoost.mh algorithm on a non-linearly separable
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using AdaBoost.
 */
TEMPLATE_TEST_CASE("WeakLearnerErrorNonLinearSepData", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;

  Row<size_t> perceptronPrediction(labels.n_cols);
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  size_t countWeakLearnerError = accu(labels != perceptronPrediction);
  eT weakLearnerErrorRate = (eT) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(inputData, labels.row(0), numClasses,
      iterations, tolerance, perceptronIter);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT error = (eT) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Iris dataset.  It
 * checks whether the Hamming loss breaches the upper bound, which is provided
 * by ztAccumulator.  This uses decision stumps as the weak learner.
 */
TEMPLATE_TEST_CASE("HammingLossIris_DS", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris_labels.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  Row<size_t> labelsvec = labels.row(0);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<ID3DecisionStump, MatType> a(tolerance);
  eT ztProduct = a.Train(inputData, labelsvec, numClasses, iterations,
      tolerance, inpBucketSize);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT hammingLoss = (eT) countError / labels.n_cols;

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
TEMPLATE_TEST_CASE("WeakLearnerErrorIris_DS", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris_labels.txt");

  // no need to map the labels here

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  Row<size_t> labelsvec = labels.row(0);

  Row<size_t> dsPrediction(labels.n_cols);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);
  ds.Classify(inputData, dsPrediction);

  size_t countWeakLearnerError = accu(labels != dsPrediction);
  eT weakLearnerErrorRate = (eT) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;

  AdaBoost<ID3DecisionStump, MatType> a(inputData, labelsvec, numClasses,
      iterations, tolerance, inpBucketSize);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT error = (eT) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It checks if the error returned by running a single instance of the
 * weak learner is close to that of a boosted weak learner using adaboost.
 * This is for the weak learner: decision stumps.
 */
TEMPLATE_TEST_CASE("HammingLossBoundVertebralColumn_DS", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  Row<size_t> labelsvec = labels.row(0);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;

  AdaBoost<ID3DecisionStump, MatType> a(tolerance);
  eT ztProduct = a.Train(inputData, labelsvec, numClasses, iterations,
      tolerance, inpBucketSize);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT hammingLoss = (eT) countError / labels.n_cols;

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
TEMPLATE_TEST_CASE("WeakLearnerErrorVertebralColumn_DS", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 3;
  const size_t inpBucketSize = 6;
  Row<size_t> dsPrediction(labels.n_cols);
  Row<size_t> labelsvec = labels.row(0);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);
  ds.Classify(inputData, dsPrediction);

  size_t countWeakLearnerError = accu(labels != dsPrediction);
  eT weakLearnerErrorRate = (eT) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<ID3DecisionStump, MatType> a(inputData, labelsvec, numClasses,
      iterations, tolerance, inpBucketSize);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT error = (eT) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on non-linearly separable
 * dataset.  It checks whether the hamming loss breaches the upperbound, which
 * is provided by ztAccumulator.  This is for the weak learner: decision stumps.
 */
TEMPLATE_TEST_CASE("HammingLossBoundNonLinearSepData_DS", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 2;
  const size_t inpBucketSize = 6;
  Row<size_t> labelsvec = labels.row(0);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);

  // Define parameters for Adaboost.
  size_t iterations = 50;
  eT tolerance = 1e-10;

  AdaBoost<ID3DecisionStump, MatType> a(tolerance);
  eT ztProduct = a.Train(inputData, labelsvec, numClasses, iterations,
      tolerance, inpBucketSize);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT hammingLoss = (eT) countError / labels.n_cols;

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
TEMPLATE_TEST_CASE("WeakLearnerErrorNonLinearSepData_DS", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  // Define your own weak learner, decision stumps in this case.
  const size_t numClasses = 2;
  const size_t inpBucketSize = 3;
  Row<size_t> labelsvec = labels.row(0);

  Row<size_t> dsPrediction(labels.n_cols);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);
  ds.Classify(inputData, dsPrediction);

  size_t countWeakLearnerError = accu(labels != dsPrediction);
  eT weakLearnerErrorRate = (eT) countWeakLearnerError / labels.n_cols;

  // Define parameters for AdaBoost.
  size_t iterations = 500;
  eT tolerance = 1e-23;

  AdaBoost<ID3DecisionStump, MatType> a(inputData, labelsvec, numClasses,
      iterations, tolerance, inpBucketSize);

  Row<size_t> predictedLabels;
  a.Classify(inputData, predictedLabels);

  size_t countError = accu(labels != predictedLabels);
  eT error = (eT) countError / labels.n_cols;

  REQUIRE(error <= weakLearnerErrorRate + 0.03);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Vertebral Column
 * dataset.  It tests the Classify function and checks for a satisfactory error
 * rate.
 */
TEMPLATE_TEST_CASE("ClassifyTest_VERTEBRALCOL", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");

  Mat<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 1000;

  MatType testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");

  Mat<size_t> trueTestLabels;

  if (!data::Load("vc2_test_labels.txt", trueTestLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  Row<size_t> perceptronPrediction(labels.n_cols);
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);
  p.Classify(inputData, perceptronPrediction);

  // Define parameters for AdaBoost.
  size_t iterations = 100;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(inputData, labels.row(0), numClasses,
      iterations, tolerance, perceptronIter);

  Row<size_t> predictedLabels1(testData.n_cols),
              predictedLabels2(testData.n_cols);
  MatType probabilities;

  a.Classify(testData, predictedLabels1);
  a.Classify(testData, predictedLabels2, probabilities);

  REQUIRE(probabilities.n_cols == testData.n_cols);
  REQUIRE(probabilities.n_rows == numClasses);

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
    REQUIRE(predictedLabels1[i] == predictedLabels2[i]);

  Col<eT> pRow;
  uword maxIndex = 0;

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
  {
    pRow = probabilities.unsafe_col(i);
    maxIndex = pRow.index_max();
    REQUIRE(predictedLabels1(i) == maxIndex);
    REQUIRE(accu(probabilities.col(i)) == Approx(1));
  }

  size_t localError = accu(trueTestLabels != predictedLabels1);
  eT lError = (eT) localError / trueTestLabels.n_cols;
  REQUIRE(lError <= 0.30);
}

/**
 * This test case runs the AdaBoost.mh algorithm on a non linearly separable
 * dataset.  It tests the Classify function and checks for a satisfactory error
 * rate.
 */
TEMPLATE_TEST_CASE("ClassifyTest_NONLINSEP", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("train_nonlinsep.txt", inputData))
    FAIL("Cannot load test dataset train_nonlinsep.txt!");

  Mat<size_t> labels;
  if (!data::Load("train_labels_nonlinsep.txt", labels))
    FAIL("Cannot load labels for train_labels_nonlinsep.txt");

  // Define your own weak learner; in this test decision stumps are used.
  const size_t numClasses = 2;
  const size_t inpBucketSize = 3;
  Row<size_t> labelsvec = labels.row(0);

  MatType testData;

  if (!data::Load("test_nonlinsep.txt", testData))
    FAIL("Cannot load test dataset test_nonlinsep.txt!");

  Mat<size_t> trueTestLabels;
  if (!data::Load("test_labels_nonlinsep.txt", trueTestLabels))
    FAIL("Cannot load labels for test_labels_nonlinsep.txt");

  Row<size_t> dsPrediction(labels.n_cols);

  ID3DecisionStump ds(inputData, labelsvec, numClasses, inpBucketSize);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<ID3DecisionStump, MatType> a(inputData, labelsvec, numClasses,
      iterations, tolerance, inpBucketSize);

  Row<size_t> predictedLabels1(testData.n_cols),
              predictedLabels2(testData.n_cols);
  MatType probabilities;

  a.Classify(testData, predictedLabels1);
  a.Classify(testData, predictedLabels2, probabilities);

  REQUIRE(probabilities.n_cols == testData.n_cols);

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
    REQUIRE(predictedLabels1[i] == predictedLabels2[i]);

  Col<eT> pRow;
  uword maxIndex = 0;

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
  {
    pRow = probabilities.unsafe_col(i);
    maxIndex = pRow.index_max();
    REQUIRE(predictedLabels1(i) == maxIndex);
    REQUIRE(accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
  }

  size_t localError = accu(trueTestLabels != predictedLabels1);
  eT lError = (eT) localError / trueTestLabels.n_cols;
  REQUIRE(lError <= 0.30);
}

/**
 * This test case runs the AdaBoost.mh algorithm on the UCI Iris Dataset.  It
 * trains it on two thirds of the Iris dataset (iris_train.csv), and tests on
 * the remaining third of the dataset (iris_test.csv).  It tests the Classify()
 * function and checks for a satisfactory error rate.
 */
TEMPLATE_TEST_CASE("ClassifyTest_IRIS", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  MatType inputData;
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load test dataset iris_train.csv!");

  Mat<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris_train_labels.csv");
  const size_t numClasses = max(labels.row(0)) + 1;

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptronIter iterations.
  size_t perceptronIter = 800;

  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);

  // Define parameters for AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(inputData, labels.row(0), numClasses,
      iterations, tolerance, perceptronIter);

  MatType testData;
  if (!data::Load("iris_test.csv", testData))
    FAIL("Cannot load test dataset iris_test.csv!");

  Row<size_t> predictedLabels(testData.n_cols);
  a.Classify(testData, predictedLabels);

  Mat<size_t> trueTestLabels;
  if (!data::Load("iris_test_labels.csv", trueTestLabels))
    FAIL("Cannot load test dataset iris_test_labels.csv!");

  Row<size_t> predictedLabels1(testData.n_cols),
              predictedLabels2(testData.n_cols);
  MatType probabilities;

  a.Classify(testData, predictedLabels1);
  a.Classify(testData, predictedLabels2, probabilities);

  REQUIRE(probabilities.n_cols == testData.n_cols);

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
    REQUIRE(predictedLabels1[i] == predictedLabels2[i]);

  Col<eT> pRow;
  uword maxIndex = 0;

  for (size_t i = 0; i < predictedLabels1.n_cols; ++i)
  {
    pRow = probabilities.unsafe_col(i);
    maxIndex = pRow.index_max();
    REQUIRE(predictedLabels1(i) == maxIndex);
    REQUIRE(accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
  }

  size_t localError = accu(trueTestLabels != predictedLabels1);
  eT lError = (eT) localError / labels.n_cols;
  REQUIRE(lError <= 0.30);
}

/**
 * Ensure that the Train() function works like it is supposed to, by building
 * AdaBoost on one dataset and then re-training on another dataset.
 */
TEMPLATE_TEST_CASE("TrainTest", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  // First train on the iris dataset.
  MatType inputData;
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load test dataset iris_train.csv!");

  Mat<size_t> labels;
  if (!data::Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris_train_labels.csv");

  const size_t numClasses = max(labels.row(0)) + 1;

  size_t perceptronIter = 800;
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  PerceptronType p(inputData, labels.row(0), numClasses, perceptronIter);

  // Now train AdaBoost.
  size_t iterations = 50;
  eT tolerance = 1e-10;
  AdaBoost<PerceptronType, MatType> a(inputData, labels.row(0), numClasses,
      iterations, tolerance, perceptronIter);

  // Now load another dataset...
  if (!data::Load("vc2.csv", inputData))
    FAIL("Cannot load test dataset vc2.csv!");
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Cannot load labels for vc2_labels.txt");

  const size_t newNumClasses = max(labels.row(0)) + 1;

  PerceptronType p2(inputData, labels.row(0), newNumClasses, perceptronIter);

  a.Train(inputData, labels.row(0), newNumClasses, iterations, tolerance,
      perceptronIter);

  // Load test set to see if it trained on vc2 correctly.
  MatType testData;
  if (!data::Load("vc2_test.csv", testData))
    FAIL("Cannot load test dataset vc2_test.csv!");

  Mat<size_t> trueTestLabels;
  if (!data::Load("vc2_test_labels.txt", trueTestLabels))
    FAIL("Cannot load labels for vc2_test_labels.txt");

  // Define parameters for AdaBoost.
  Row<size_t> predictedLabels(testData.n_cols);
  a.Classify(testData, predictedLabels);

  size_t localError = accu(trueTestLabels != predictedLabels);
  eT lError = (eT) localError / trueTestLabels.n_cols;

  REQUIRE(lError <= 0.30);
}

TEMPLATE_TEST_CASE("PerceptronSerializationTest", "[AdaBoostTest]", fmat, mat)
{
  using MatType = TestType;

  // Build an AdaBoost object.
  MatType data = randu<MatType>(10, 500);
  Row<size_t> labels(500);
  for (size_t i = 0; i < 250; ++i)
    labels[i] = 0;
  for (size_t i = 250; i < 500; ++i)
    labels[i] = 1;

  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  AdaBoost<PerceptronType, MatType> ab(data, labels, 2, 50, 1e-10, 800);

  // Now create another dataset to train with.
  MatType otherData = randu<MatType>(5, 200);
  Row<size_t> otherLabels(200);
  for (size_t i = 0; i < 100; ++i)
    otherLabels[i] = 1;
  for (size_t i = 100; i < 150; ++i)
    otherLabels[i] = 0;
  for (size_t i = 150; i < 200; ++i)
    otherLabels[i] = 2;

  AdaBoost<PerceptronType, MatType> abText(otherData, otherLabels, 3, 50, 1e-10,
      500);

  AdaBoost<PerceptronType, MatType> abXml, abBinary;

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

TEMPLATE_TEST_CASE("ID3DecisionStumpSerializationTest", "[AdaBoostTest]", mat,
    fmat)
{
  using MatType = TestType;

  // Build an AdaBoost object.
  MatType data = randu<MatType>(10, 500);
  Row<size_t> labels(500);
  for (size_t i = 0; i < 250; ++i)
    labels[i] = 0;
  for (size_t i = 250; i < 500; ++i)
    labels[i] = 1;

  AdaBoost<ID3DecisionStump, MatType> ab(data, labels, 2, 50, 1e-10, 40);

  // Now create another dataset to train with.
  MatType otherData = randu<MatType>(5, 200);
  Row<size_t> otherLabels(200);
  for (size_t i = 0; i < 100; ++i)
    otherLabels[i] = 1;
  for (size_t i = 100; i < 150; ++i)
    otherLabels[i] = 0;
  for (size_t i = 150; i < 200; ++i)
    otherLabels[i] = 2;

  AdaBoost<ID3DecisionStump, MatType> abText(otherData, otherLabels, 3, 50,
      1e-10, 25);

  AdaBoost<ID3DecisionStump, MatType> abXml, abBinary;

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

TEMPLATE_TEST_CASE("AdaBoostSinglePointClassify", "[AdaBoostTest]", mat, fmat)
{
  using MatType = TestType;

  // Create random data.
  MatType data = randu<MatType>(10, 100);
  // Create random labels.
  Row<size_t> labels = randi<Row<size_t>>(100, distr_param(0, 3));

  // Train a model.
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  AdaBoost<PerceptronType, MatType> ab(data, labels, 4);

  // Ensure that we can get single-point classifications.
  for (size_t i = 0; i < 100; ++i)
  {
    const size_t prediction = ab.Classify(data.col(i));

    REQUIRE(prediction <= 3);
  }
}

TEMPLATE_TEST_CASE("AdaBoostSinglePointClassifyWithProbs", "[AdaBoostTest]",
    mat, fmat)
{
  using MatType = TestType;
  using eT = typename MatType::elem_type;

  // Create random data.
  MatType data = randu<MatType>(10, 100);
  // Create random labels.
  Row<size_t> labels = randi<Row<size_t>>(100, distr_param(0, 3));

  // Train a model.
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  AdaBoost<PerceptronType, MatType> ab(data, labels, 4);

  // Ensure that we can get single-point classifications.
  for (size_t i = 0; i < 100; ++i)
  {
    size_t prediction;
    Row<eT> probabilities;
    ab.Classify(data.col(i), prediction, probabilities);

    REQUIRE(prediction <= 3);
    REQUIRE(accu(probabilities) == Approx((eT) 1.0));
  }
}

// Make sure that everything works when we use the constructor that takes extra
// hyperparameters.
TEMPLATE_TEST_CASE("AdaBoostParamsConstructor", "[AdaBoostTest]", fmat, mat)
{
  using MatType = TestType;

  MatType inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Cannot load test dataset iris.csv!");

  Mat<size_t> labels;
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load labels for iris iris_labels.txt");

  const size_t numClasses = max(labels.row(0)) + 1;

  // Create two AdaBoost models.  One does not allow the perceptron to train for
  // more than one iteration, and therefore should get less accuracy than the
  // one we let train in full.
  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;

  AdaBoost<PerceptronType, MatType> a1(inputData, labels, numClasses, 2, 1e-6,
      1 /* perceptron max iterations */);
  AdaBoost<PerceptronType, MatType> a2(inputData, labels, numClasses, 2, 1e-6,
      100 /* perceptron max iterations */);

  // Make sure test data performance is better for a2.
  Row<size_t> predictions1, predictions2;
  a1.Classify(inputData, predictions1);
  a2.Classify(inputData, predictions2);

  const size_t correct1 = accu(predictions1 == labels);
  const size_t correct2 = accu(predictions2 == labels);

  REQUIRE(correct2 > 0);
  REQUIRE(correct2 >= correct1);
}

// Ensure that all Train() overloads work correctly.
TEMPLATE_TEST_CASE("AdaBoostTrainOverloads", "[AdaBoostTest]", fmat, mat)
{
  using MatType = TestType;

  // Create random data.
  MatType data = randu<MatType>(10, 100);
  // Create random labels.
  Row<size_t> labels = randi<Row<size_t>>(100, distr_param(0, 3));

  using PerceptronType =
      Perceptron<SimpleWeightUpdate, ZeroInitialization, MatType>;
  AdaBoost<PerceptronType, MatType> a1, a2, a3, a4;
  a1.MaxIterations() = 65;
  a1.Tolerance() = 2e-4;
  a2.Tolerance() = 2e-5;

  a1.Train(data, labels, 4);
  a2.Train(data, labels, 4, 15);
  a3.Train(data, labels, 4, 55, 1e-3);
  a4.Train(data, labels, 4, 60, 2e-3, 100);

  // Make sure hyperparameters were set correctly, where appropriate.
  REQUIRE(a1.MaxIterations() == 65);
  REQUIRE(a2.MaxIterations() == 15);
  REQUIRE(a3.MaxIterations() == 55);
  REQUIRE(a4.MaxIterations() == 60);

  REQUIRE(a1.Tolerance() == Approx(2e-4));
  REQUIRE(a2.Tolerance() == Approx(2e-5));
  REQUIRE(a3.Tolerance() == Approx(1e-3));
  REQUIRE(a4.Tolerance() == Approx(2e-3));

  // Make sure anything at all was trained.
  REQUIRE(a1.WeakLearners() > 0);
  REQUIRE(a2.WeakLearners() > 0);
  REQUIRE(a3.WeakLearners() > 0);
  REQUIRE(a4.WeakLearners() > 0);

  // Make sure the maximum number of iterations in the perceptron was set
  // properly.
  REQUIRE(a1.WeakLearner(0).MaxIterations() == 1000);
  REQUIRE(a2.WeakLearner(0).MaxIterations() == 1000);
  REQUIRE(a3.WeakLearner(0).MaxIterations() == 1000);
  REQUIRE(a4.WeakLearner(0).MaxIterations() == 100);
}
