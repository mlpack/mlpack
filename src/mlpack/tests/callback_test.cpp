/**
 * @file tests/callback_test.cpp
 *
 * Test the Linear SVM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>
#include <mlpack/methods/lmnn.hpp>
#include <mlpack/methods/logistic_regression.hpp>
#include <mlpack/methods/nca.hpp>
#include <mlpack/methods/softmax_regression.hpp>
#include <mlpack/methods/sparse_autoencoder.hpp>

#include "catch.hpp"

using namespace mlpack;

/**
 * Test a FFN model with PrintLoss callback.
 */
TEST_CASE("FFNCallbackTest", "[CallbackTest]")
{
  arma::mat data;
  arma::mat labels;

  if (!data::Load("lab1.csv", data))
    FAIL("Cannot load test dataset lab1.csv!");
  if (!data::Load("lab3.csv", labels))
    FAIL("Cannot load test dataset lab3.csv!");

  FFN<MeanSquaredError, RandomInitialization> model;

  model.Add<Linear>(2);
  model.Add<Sigmoid>();
  model.Add<Linear>(1);
  model.Add<Sigmoid>();

  std::stringstream stream;
  model.Train(data, labels, ens::PrintLoss(stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Test a FFN model with PrintLoss callback and optimizer parameter.
 */
TEST_CASE("FFNWithOptimizerCallbackTest", "[CallbackTest]")
{
  arma::mat data;
  arma::mat labels;

  if (!data::Load("lab1.csv", data))
    FAIL("Cannot load test dataset lab1.csv!");
  if (!data::Load("lab3.csv", labels))
    FAIL("Cannot load test dataset lab3.csv!");

  FFN<MeanSquaredError, RandomInitialization> model;

  model.Add<Linear>(2);
  model.Add<Sigmoid>();
  model.Add<Linear>(1);
  model.Add<Sigmoid>();

  std::stringstream stream;
  ens::StandardSGD opt(0.1, 1, 5);
  model.Train(data, labels, opt, ens::PrintLoss(stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Test a RNN model with PrintLoss callback.
 */
TEST_CASE("RNNCallbackTest", "[CallbackTest]")
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::zeros(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood, RandomInitialization> model(
      rho, false, NegativeLogLikelihood(), init);
  model.Add<Linear>(10);

  // Use LSTM layer with 3 units.
  model.Add<LSTM>(3);
  model.Add<LogSoftMax>();

  std::stringstream stream;
  model.Train(input, target, ens::PrintLoss(stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Test a RNN model with PrintLoss callback and optimizer parameter.
 */
TEST_CASE("RNNWithOptimizerCallbackTest", "[CallbackTest]")
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::zeros(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood, RandomInitialization> model(
      rho, false, NegativeLogLikelihood(), init);
  model.Add<Linear>(10);

  // Use LSTM layer with 3 units.
  model.Add<LSTM>(3);
  model.Add<LogSoftMax>();

  std::stringstream stream;
  ens::StandardSGD opt(0.1, 1, 5);
  model.Train(input, target, opt, ens::PrintLoss(stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Test Logistic regression implementation with PrintLoss callback.
 */
TEST_CASE("LRWithOptimizerCallback", "[CallbackTest]")
{
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  ens::StandardSGD sgd(0.1, 1, 5);
  LogisticRegression<> logisticRegression(data, responses, sgd, 0.001);
  std::stringstream stream;
  logisticRegression.Train<ens::StandardSGD>(data, responses, sgd,
                                             ens::PrintLoss(stream));

  REQUIRE(stream.str().length() > 0);
}

/**
 * Test LMNN implementation with ProgressBar callback.
 */
TEST_CASE("LMNNWithOptimizerCallback", "[CallbackTest]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                      " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNN<> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  std::stringstream stream;

  lmnn.LearnDistance(outputMatrix, ens::ProgressBar(70, stream));
  REQUIRE(stream.str().length() > 0);
}

/**
 * Test NCA implementation with ProgressBar callback.
 */
TEST_CASE("NCAWithOptimizerCallback", "[CallbackTest]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                   " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  NCA<SquaredEuclideanDistance> nca(data, labels);

  arma::mat outputMatrix;
  std::stringstream stream;

  nca.LearnDistance(outputMatrix, ens::ProgressBar(70, stream));
  REQUIRE(stream.str().length() > 0);
}

/**
 * Test softmax_regression implementation with PrintLoss callback.
 */
TEST_CASE("SRWithOptimizerCallback", "[CallbackTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 3;
  const size_t numClasses = 3;
  const double lambda = 0.5;

  // Generate two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 2; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 2; i < points; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  ens::StandardSGD sgd(0.1, 1, 5);
  std::stringstream stream;
  // Train softmax regression object.
  SoftmaxRegression sr(data, labels, numClasses, lambda);
  sr.Train(data, labels, numClasses, sgd, ens::ProgressBar(70, stream));

  REQUIRE(stream.str().length() > 0);
}

/*
 * Tests the RBM Implementation with PrintLoss callback.
 *
TEST_CASE("RBMCallbackTest", "[CallbackTest]")
{
  // Normalised dataset.
  int hiddenLayerSize = 10;
  size_t batchSize = 10;
  arma::mat trainData, testData, dataset;
  arma::mat trainLabelsTemp, testLabelsTemp;
  trainData.load("digits_train.arm");

  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization> model(trainData,
                                    gaussian,
                                    trainData.n_rows,
                                    hiddenLayerSize,
                                    batchSize);

  size_t numRBMIterations = 10;
  ens::StandardSGD msgd(0.03, batchSize, numRBMIterations, 0, true);
  std::stringstream stream;

  // Call the train function with printloss callback.
  double objVal = model.Train(msgd, ens::ProgressBar(70, stream));
  REQUIRE(!std::isnan(objVal));
  REQUIRE(stream.str().length() > 0);
}*/

/**
 * Tests the SparseAutoencoder implementation with
 * StoreBestCoordinates callback.
 */
TEST_CASE("SparseAutoencodeCallbackTest", "[CallbackTest]")
{
  // Simple fake dataset.
  arma::mat data1("0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5");

  ens::L_BFGS optimizer(5, 100);
  ens::StoreBestCoordinates<arma::mat> cb;
  SparseAutoencoder encoder2(data1, 5, 1, 0, 0, 0 , optimizer, cb);
  REQUIRE(cb.BestObjective() > 0);
}
