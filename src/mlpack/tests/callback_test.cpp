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
#include <ensmallen.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/brnn.hpp>
#include <mlpack/methods/ann/rbm/rbm.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/lmnn/lmnn.hpp>
#include <mlpack/methods/nca/nca.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_impl.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/sparse_autoencoder/sparse_autoencoder.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::regression;
using namespace mlpack::lmnn;
using namespace mlpack::metric;
using namespace mlpack::nca;
using namespace mlpack::distribution;

/**
 * Construct a 2-class dataset out of noisy sines.
 *
 * @param data Input data used to store the noisy sines.
 * @param labels Labels used to store the target class of the noisy sines.
 * @param points Number of points/features in a single sequence.
 * @param sequences Number of sequences for each class.
 * @param noise The noise factor that influences the sines.
 */
void GenerateNoisySines(arma::cube& data,
                        arma::mat& labels,
                        const size_t points,
                        const size_t sequences,
                        const double noise = 0.3)
{
  arma::colvec x =  arma::linspace<arma::colvec>(0, points - 1, points) /
      points * 20.0;
  arma::colvec y1 = arma::sin(x + arma::as_scalar(arma::randu(1)) * 3.0);
  arma::colvec y2 = arma::sin(x / 2.0 + arma::as_scalar(arma::randu(1)) * 3.0);

  data = arma::zeros(1 /* single dimension */, sequences * 2, points);
  labels = arma::zeros(2 /* 2 classes */, sequences * 2);

  for (size_t seq = 0; seq < sequences; seq++)
  {
    arma::vec sequence = arma::randu(points) * noise + y1 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    for (size_t i = 0; i < points; ++i)
      data(0, seq, i) = sequence[i];

    labels(0, seq) = 1;

    sequence = arma::randu(points) * noise + y2 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    for (size_t i = 0; i < points; ++i)
      data(0, sequences + seq, i) = sequence[i];

    labels(1, sequences + seq) = 1;
  }
}

/**
 * Test a FFN model with PrintLoss callback.
 */
TEST_CASE("FFNCallbackTest", "[CallbackTest]")
{
  arma::mat data;
  arma::mat labels;

  data::Load("lab1.csv", data, true);
  data::Load("lab3.csv", labels, true);

  FFN<MeanSquaredError<>, RandomInitialization> model;

  model.Add<Linear<>>(1, 2);
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>(2, 1);
  model.Add<SigmoidLayer<>>();

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

  data::Load("lab1.csv", data, true);
  data::Load("lab3.csv", labels, true);

  FFN<MeanSquaredError<>, RandomInitialization> model;

  model.Add<Linear<>>(1, 2);
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>(2, 1);
  model.Add<SigmoidLayer<>>();

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
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> model(
      rho, false, NegativeLogLikelihood<>(), init);
  model.Add<IdentityLayer<>>();
  model.Add<Linear<>>(1, 10);

  // Use LSTM layer with rho.
  model.Add<LSTM<>>(10, 3, rho);
  model.Add<LogSoftMax<>>();

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
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> model(
      rho, false, NegativeLogLikelihood<>(), init);
  model.Add<IdentityLayer<>>();
  model.Add<Linear<>>(1, 10);

  // Use LSTM layer with rho.
  model.Add<LSTM<>>(10, 3, rho);
  model.Add<LogSoftMax<>>();

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
 */
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
}

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
  mlpack::nn::SparseAutoencoder encoder2(data1, 5, 1, 0, 0, 0 , optimizer, cb);
  REQUIRE(cb.BestObjective() > 0);
}

TEST_CASE("BRNNWithOptimizerCallbackTest", "[CallbackTest]")
{
  const size_t rho = 10;

  arma::cube input;
  arma::mat labelsTemp;
  GenerateNoisySines(input, labelsTemp, rho, 6);

  arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
    labels.tube(0, i).fill(value);
  }

  Add<> add(4);
  Linear<> lookup(1, 4);
  SigmoidLayer<> sigmoidLayer;
  Linear<> linear(4, 4);
  Recurrent<>* recurrent = new Recurrent<>(
      add, lookup, linear, sigmoidLayer, rho);

  BRNN<> model(rho);
  model.Add<IdentityLayer<> >();
  model.Add(recurrent);
  model.Add<Linear<> >(4, 5);
  std::stringstream stream;

  ens::StandardSGD opt(0.1, 1, 500 * input.n_cols, -100);
  model.Train(input, labels, opt, ens::PrintLoss(stream));

  REQUIRE(stream.str().length() > 0);
}

TEST_CASE("BRNNCallbackTest", "[CallbackTest]")
{
  const size_t rho = 10;

  arma::cube input;
  arma::mat labelsTemp;
  GenerateNoisySines(input, labelsTemp, rho, 6);

  arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
    labels.tube(0, i).fill(value);
  }

  Add<> add(4);
  Linear<> lookup(1, 4);
  SigmoidLayer<> sigmoidLayer;
  Linear<> linear(4, 4);
  Recurrent<>* recurrent = new Recurrent<>(
      add, lookup, linear, sigmoidLayer, rho);

  BRNN<> model(rho);
  model.Add<IdentityLayer<> >();
  model.Add(recurrent);
  model.Add<Linear<> >(4, 5);
  std::stringstream stream;

  model.Train(input, labels, ens::PrintLoss(stream));
  REQUIRE(stream.str().length() > 0);
}
