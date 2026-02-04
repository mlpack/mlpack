/**
 * @file tests/ann/fp16_test.cpp
 * @author Ryan Curtin
 *
 * Test that mlpack neural networks successfully run with fp16 precision.  This
 * also instantiates all layers for serialization and therefore serves as a
 * check for any compilation warnings when using fp16.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#ifdef ARMA_HAVE_FP16

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include "ann_test_tools.hpp"

// Register serialization for all fp16 layers.
CEREAL_REGISTER_MLPACK_LAYERS(arma::hmat);

TEST_CASE("FFNSimpleFP16Test", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::hmat trainData;
  if (!Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::hmat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);
  trainLabels -= 1; // Labels should be from 0 to numClasses - 1.

  arma::hmat testData;
  if (!Load("thyroid_test.csv", testData))
    FAIL("Cannot load dataset thyroid_test.csv");

  arma::hmat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);
  testLabels -= 1; // Labels should be from 0 to numClasses - 1.

  FFN<NegativeLogLikelihoodType<arma::hmat>, RandomInitialization, arma::hmat>
      model;
  model.Add<Linear<arma::hmat>>(8);
  model.Add<Sigmoid<arma::hmat>>();
  model.Add<Dropout<arma::hmat>>();
  model.Add<Linear<arma::hmat>>(3);
  model.Add<LogSoftMax<arma::hmat>>();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  const double error = TestClassificationNetwork(model, trainData, trainLabels,
      testData, testLabels, 10);
  REQUIRE(error <= 0.1);
}

TEST_CASE("RNNSimpleFP16Test", "[RecurrentNetworkTest]")
{
  const double err =
      ImpulseStepDataTest<LinearRecurrent<arma::hmat>, arma::fp16>(1, 5);
  REQUIRE(err <= 0.005);
}

#endif
