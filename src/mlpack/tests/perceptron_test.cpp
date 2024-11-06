/**
 * @file tests/perceptron_test.cpp
 * @author Udit Saxena
 *
 * Tests for perceptron.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace arma;

/**
 * This test tests whether the SimpleWeightUpdate updates weights and biases correctly,
 * without specifying the instance weight.
 */
TEST_CASE("SimpleWeightUpdateWeights", "[PerceptronTest]")
{
  SimpleWeightUpdate wip;

  /**
   * The weights of the incorrectly classified class should decrease while the
   * weight of the correct class should increase.
   */
  vec trainingPoint("1 2 3 4 5");
  mat weights("0 1 6;"
              "2 3 6;"
              "4 5 6;"
              "6 7 6;"
              "8 9 6");
  vec biases("2 5 7");
  size_t incorrectClass = 0;
  size_t correctClass = 2;

  wip.UpdateWeights(trainingPoint, weights, biases, incorrectClass,
                    correctClass);

  REQUIRE(weights(0, 0) == -1);
  REQUIRE(weights(1, 0) == 0);
  REQUIRE(weights(2, 0) == 1);
  REQUIRE(weights(3, 0) == 2);
  REQUIRE(weights(4, 0) == 3);

  REQUIRE(weights(0, 2) == 7);
  REQUIRE(weights(1, 2) == 8);
  REQUIRE(weights(2, 2) == 9);
  REQUIRE(weights(3, 2) == 10);
  REQUIRE(weights(4, 2) == 11);

  REQUIRE(biases(0) == 1);
  REQUIRE(biases(2) == 8);
}

/**
 * This test tests whether the SimpleWeightUpdate updates weights and biases correctly,
 * and specifies the instance weight.
 */
TEST_CASE("SimpleWeightUpdateInstanceWeight", "[PerceptronTest]")
{
  SimpleWeightUpdate wip;

  /**
   * The weights of the incorrectly classified class should decrease
   * while the weights of the correct class should increase.  The decrease and
   * increase depend on the specified instance weight.
   */
  vec trainingPoint("1 2 3 4 5");
  mat weights("0 1 6;"
              "2 3 6;"
              "4 5 6;"
              "6 7 6;"
              "8 9 6");
  vec biases("2 5 7");
  size_t incorrectClass = 0;
  size_t correctClass = 2;
  double instanceWeight = 3.0;

  wip.UpdateWeights(trainingPoint, weights, biases, incorrectClass,
                    correctClass, instanceWeight);

  REQUIRE(weights(0, 0) == -3);
  REQUIRE(weights(1, 0) == -4);
  REQUIRE(weights(2, 0) == -5);
  REQUIRE(weights(3, 0) == -6);
  REQUIRE(weights(4, 0) == -7);

  REQUIRE(weights(0, 2) == 9);
  REQUIRE(weights(1, 2) == 12);
  REQUIRE(weights(2, 2) == 15);
  REQUIRE(weights(3, 2) == 18);
  REQUIRE(weights(4, 2) == 21);

  REQUIRE(biases(0) == -1);
  REQUIRE(biases(2) == 10);
}

/**
 * This test tests whether the perceptron converges for the AND gate classifier.
 */
TEST_CASE("And", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 0, 1, 1, 0 },
                { 1, 0, 1, 0 } };
  Mat<size_t> labels;
  labels = { 0, 0, 1, 0 };

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData = { { 0, 1, 1, 0 },
               { 1, 0, 1, 0 } };
  Row<size_t> predictedLabels;
  p.Classify(testData, predictedLabels);

  REQUIRE(predictedLabels(0) == 0);
  REQUIRE(predictedLabels(1) == 0);
  REQUIRE(predictedLabels(2) == 1);
  REQUIRE(predictedLabels(3) == 0);

  // Test single-point classify too.
  predictedLabels(0) = p.Classify(testData.col(0));
  predictedLabels(1) = p.Classify(testData.col(1));
  predictedLabels(2) = p.Classify(testData.col(2));
  predictedLabels(3) = p.Classify(testData.col(3));

  REQUIRE(predictedLabels(0) == 0);
  REQUIRE(predictedLabels(1) == 0);
  REQUIRE(predictedLabels(2) == 1);
  REQUIRE(predictedLabels(3) == 0);
}

/**
 * This test tests whether the perceptron converges for the OR gate classifier.
 */
TEST_CASE("Or", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 0, 1, 1, 0 },
                { 1, 0, 1, 0 } };

  Mat<size_t> labels;
  labels = { 1, 1, 1, 0 };

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData = { { 0, 1, 1, 0 },
               { 1, 0, 1, 0 } };
  Row<size_t> predictedLabels;
  p.Classify(testData, predictedLabels);

  REQUIRE(predictedLabels(0, 0) == 1);
  REQUIRE(predictedLabels(0, 1) == 1);
  REQUIRE(predictedLabels(0, 2) == 1);
  REQUIRE(predictedLabels(0, 3) == 0);
}

/**
 * This tests the convergence on a set of linearly separable data with 3
 * classes.
 */
TEST_CASE("Random3", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 0, 1, 1, 4, 5, 4, 1, 2, 1 },
                { 1, 0, 1, 1, 1, 2, 4, 5, 4 } };

  Mat<size_t> labels;
  labels = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };

  Perceptron<> p(trainData, labels.row(0), 3, 1000);

  mat testData;
  testData = { { 0, 1, 1 },
               { 1, 0, 1 } };
  Row<size_t> predictedLabels;
  p.Classify(testData, predictedLabels);

  for (size_t i = 0; i < predictedLabels.n_cols; ++i)
    REQUIRE(predictedLabels(0, i) == 0);
}

/**
 * This tests the convergence of the perceptron on a dataset which has only TWO
 * points which belong to different classes.
 */
TEST_CASE("TwoPoints", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 0, 1 },
                { 1, 0 } };

  Mat<size_t> labels;
  labels = { 0, 1 };

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData = { { 0, 1 },
               { 1, 0 } };
  Row<size_t> predictedLabels;
  p.Classify(testData, predictedLabels);

  REQUIRE(predictedLabels(0, 0) == 0);
  REQUIRE(predictedLabels(0, 1) == 1);
}

/**
 * This tests the convergence of the perceptron on a dataset which has a
 * non-linearly separable dataset.  We test on multiple element types to ensure
 * that MatType can be set correctly.
 */
TEMPLATE_TEST_CASE("NonLinearlySeparableDataset", "[PerceptronTest]", float,
    double)
{
  using eT = TestType;

  Mat<eT> trainData;
  trainData = { { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 } };

  Mat<size_t> labels;
  labels = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1 };

  Perceptron<SimpleWeightUpdate, ZeroInitialization, Mat<eT>> p(trainData,
      labels.row(0), 2, 1000);

  Mat<eT> testData;
  testData = { { 3,   4,   5,   6 },
               { 3, 2.3, 1.7, 1.5 } };
  Row<size_t> predictedLabels;
  p.Classify(testData, predictedLabels);

  REQUIRE(predictedLabels(0, 0) == 0);
  REQUIRE(predictedLabels(0, 1) == 0);
  REQUIRE(predictedLabels(0, 2) == 1);
  REQUIRE(predictedLabels(0, 3) == 1);
}

TEST_CASE("SecondaryConstructor", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 } };

  Mat<size_t> labels;
  labels = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1 };

  Perceptron<> p1(trainData, labels.row(0), 2, 1000);

  Perceptron<> p2(p1);

  REQUIRE(p1.Weights().n_elem > 0);
  REQUIRE(p2.Weights().n_elem > 0);
}

/**
 * This tests that we can build the Perceptron when specifying instance weights.
 */
TEST_CASE("InstanceWeightsConstructor", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 } };

  Mat<size_t> labels;
  labels = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1 };

  rowvec instanceWeights;
  instanceWeights = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9,
      0.8, 0.7, 0.6, 0.5, 0.4 };

  Perceptron<> p(trainData, labels.row(0), 2, instanceWeights, 1000);

  REQUIRE(p.Weights().n_elem > 0);
}

/**
 * This tests that incremental training can be stopped with `Reset()`.
 */
TEST_CASE("IncrementalTrainingTest", "[PerceptronTest]")
{
  mat trainData = randu<mat>(10, 1000);
  for (size_t i = 0; i < 500; ++i)
    trainData.col(i) += 0.6;
  for (size_t i = 500; i < 1000; ++i)
    trainData.col(i) += 0.8;
  Row<size_t> labels(1000);
  labels.subvec(0, 499).zeros();
  labels.subvec(500, 999).ones();

  Perceptron<> p1, p2;

  // This should result in the same model, because the default initialization is
  // zeros.
  p1.Train(trainData, labels, 2, 1);
  p2.Train(trainData, labels, 2, 1);

  REQUIRE(approx_equal(p1.Weights(), p2.Weights(), "absdiff", 1e-5));
  REQUIRE(approx_equal(p1.Biases(), p2.Biases(), "absdiff", 1e-5));

  // Resetting and retraining p2 should result in the same model.
  p2.Reset();
  p2.Train(trainData, labels, 2);

  REQUIRE(approx_equal(p1.Weights(), p2.Weights(), "absdiff", 1e-5));
  REQUIRE(approx_equal(p1.Biases(), p2.Biases(), "absdiff", 1e-5));

  // Training p1 again should result in a different model.
  p1.Train(trainData, labels, 2);

  // The biases often converge to the same -1/0/1 values, but it's sufficient
  // to just check the weights.
  REQUIRE(!approx_equal(p1.Weights(), p2.Weights(), "absdiff", 1e-5));
}

// Test all forms of Train().
TEST_CASE("TrainFormTest", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 } };

  Row<size_t> labels;
  labels = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1 };

  rowvec weights(labels.n_elem);
  weights.ones();

  Perceptron<> p1(trainData, labels, 2);
  Perceptron<> p2(trainData, labels, 2, weights);
  Perceptron<> p3, p4, p5, p6;
  p3.Train(trainData, labels, 2);
  p4.Train(trainData, labels, 2, 50);
  p5.Train(trainData, labels, 2, weights);
  p6.Train(trainData, labels, 2, weights, 50);

  mat testData;
  testData = { { 3,   4,   5,   6 },
               { 3, 2.3, 1.7, 1.5 } };
  Row<size_t> predictions1, predictions2, predictions3, predictions4,
      predictions5, predictions6;
  Row<size_t> trueLabels = { 0, 0, 1, 1 };
  p1.Classify(testData, predictions1);
  p2.Classify(testData, predictions2);
  p3.Classify(testData, predictions3);
  p4.Classify(testData, predictions4);
  p5.Classify(testData, predictions5);
  p6.Classify(testData, predictions6);

  REQUIRE(all(predictions1 == trueLabels));
  REQUIRE(all(predictions2 == trueLabels));
  REQUIRE(all(predictions3 == trueLabels));
  REQUIRE(all(predictions4 == trueLabels));
  REQUIRE(all(predictions5 == trueLabels));
  REQUIRE(all(predictions6 == trueLabels));
}
