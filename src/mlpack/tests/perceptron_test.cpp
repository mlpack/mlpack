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

  CHECK(weights(0, 0) == -1);
  CHECK(weights(1, 0) == 0);
  CHECK(weights(2, 0) == 1);
  CHECK(weights(3, 0) == 2);
  CHECK(weights(4, 0) == 3);

  CHECK(weights(0, 2) == 7);
  CHECK(weights(1, 2) == 8);
  CHECK(weights(2, 2) == 9);
  CHECK(weights(3, 2) == 10);
  CHECK(weights(4, 2) == 11);

  CHECK(biases(0) == 1);
  CHECK(biases(2) == 8);
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

  CHECK(weights(0, 0) == -3);
  CHECK(weights(1, 0) == -4);
  CHECK(weights(2, 0) == -5);
  CHECK(weights(3, 0) == -6);
  CHECK(weights(4, 0) == -7);

  CHECK(weights(0, 2) == 9);
  CHECK(weights(1, 2) == 12);
  CHECK(weights(2, 2) == 15);
  CHECK(weights(3, 2) == 18);
  CHECK(weights(4, 2) == 21);

  CHECK(biases(0) == -1);
  CHECK(biases(2) == 10);
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

  CHECK(predictedLabels(0, 0) == 0);
  CHECK(predictedLabels(0, 1) == 0);
  CHECK(predictedLabels(0, 2) == 1);
  CHECK(predictedLabels(0, 3) == 0);
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

  CHECK(predictedLabels(0, 0) == 1);
  CHECK(predictedLabels(0, 1) == 1);
  CHECK(predictedLabels(0, 2) == 1);
  CHECK(predictedLabels(0, 3) == 0);
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
    CHECK(predictedLabels(0, i) == 0);
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

  CHECK(predictedLabels(0, 0) == 0);
  CHECK(predictedLabels(0, 1) == 1);
}

/**
 * This tests the convergence of the perceptron on a dataset which has a
 * non-linearly separable dataset.
 */
TEST_CASE("NonLinearlySeparableDataset", "[PerceptronTest]")
{
  mat trainData;
  trainData = { { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 } };

  Mat<size_t> labels;
  labels = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1 };

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData = { { 3,   4,   5,   6 },
               { 3, 2.3, 1.7, 1.5 } };
  Row<size_t> predictedLabels;
  p.Classify(testData, predictedLabels);

  CHECK(predictedLabels(0, 0) == 0);
  CHECK(predictedLabels(0, 1) == 0);
  CHECK(predictedLabels(0, 2) == 1);
  CHECK(predictedLabels(0, 3) == 1);
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
