/**
 * @file feedforward_network_test.cpp
 * @author Marcus Edel
 * @author Palash Ahuja
 *
 * Tests the feed forward network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(FeedForwardNetworkTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<typename MatType = arma::mat>
void BuildVanillaNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
                         const size_t outputSize,
                         const size_t hiddenLayerSize,
                         const size_t maxEpochs,
                         const double classificationErrorThreshold)
{
  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Output
   *  Layer         Layer         Layer
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |     +>|     |     +>|     |
   * +-----+     | +--+--+     | +-----+
   *             |             |
   *  Bias       |  Bias       |
   *  Layer      |  Layer      |
   * +-----+     | +-----+     |
   * |     |     | |     |     |
   * |     +-----+ |     +-----+
   * |     |       |     |
   * +-----+       +-----+
   */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  RMSprop<decltype(model)> opt(model, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, -1);

  model.Train(std::move(trainData), std::move(trainLabels), opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (int(arma::as_scalar(prediction.col(i))) ==
        int(arma::as_scalar(testLabels.col(i))))
    {
      error++;
    }
  }

  double classificationError = 1 - double(error) / testData.n_cols;
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildVanillaNetwork<>
      (trainData, trainLabels, testData, testLabels, 3, 8, 70, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  // Vanilla neural net with logistic activation function.
  BuildVanillaNetwork<>
      (dataset, labels, dataset, labels, 2, 10, 50, 0.2);
}

/**
 * Train and evaluate a Dropout network with the specified structure.
 */
template<typename MatType = arma::mat>
void BuildDropoutNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
                         const size_t outputSize,
                         const size_t hiddenLayerSize,
                         const size_t maxEpochs,
                         const double classificationErrorThreshold)
{
  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Dropout      Output
   *  Layer         Layer         Layer        Layer
   * +-----+       +-----+       +-----+       +-----+
   * |     |       |     |       |     |       |     |
   * |     +------>|     +------>|     +------>|     |
   * |     |     +>|     |       |     |       |     |
   * +-----+     | +--+--+       +-----+       +-----+
   *             |
   *  Bias       |
   *  Layer      |
   * +-----+     |
   * |     |     |
   * |     +-----+
   * |     |
   * +-----+
   */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  RMSprop<decltype(model)> opt(model, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, -1);

  model.Train(std::move(trainData), std::move(trainLabels), opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (int(arma::as_scalar(prediction.col(i))) ==
        int(arma::as_scalar(testLabels.col(i))))
    {
      error++;
    }
  }

  double classificationError = 1 - double(error) / testData.n_cols;
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train the dropout network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(DropoutNetworkTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildDropoutNetwork<>
      (trainData, trainLabels, testData, testLabels, 3, 8, 70, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  // Vanilla neural net with logistic activation function.
  BuildDropoutNetwork<>
      (dataset, labels, dataset, labels, 2, 10, 50, 0.2);
}

/**
 * Train and evaluate a DropConnect network(with a baselayer) with the
 * specified structure.
 */
template<typename MatType = arma::mat>
void BuildDropConnectNetwork(MatType& trainData,
                             MatType& trainLabels,
                             MatType& testData,
                             MatType& testLabels,
                             const size_t outputSize,
                             const size_t hiddenLayerSize,
                             const size_t maxEpochs,
                             const double classificationErrorThreshold)
{
 /*
  *  Construct a feed forward network with trainData.n_rows input nodes,
  *  hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
  *  network struct that looks like:
  *
  *  Input         Hidden     DropConnect     Output
  *  Layer         Layer         Layer        Layer
  * +-----+       +-----+       +-----+       +-----+
  * |     |       |     |       |     |       |     |
  * |     +------>|     +------>|     +------>|     |
  * |     |     +>|     |       |     |       |     |
  * +-----+     | +--+--+       +-----+       +-----+
  *             |
  *  Bias       |
  *  Layer      |
  * +-----+     |
  * |     |     |
  * |     +-----+
  * |     |
  * +-----+
  *
  *
  */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<DropConnect<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  RMSprop<decltype(model)> opt(model, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, -1);

  model.Train(std::move(trainData), std::move(trainLabels), opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (int(arma::as_scalar(prediction.col(i))) ==
        int(arma::as_scalar(testLabels.col(i))))
    {
      error++;
    }
  }

  double classificationError = 1 - double(error) / testData.n_cols;
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train the dropconnect network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(DropConnectNetworkTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildDropConnectNetwork<>
      (trainData, trainLabels, testData, testLabels, 3, 8, 70, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  // Vanilla neural net with logistic activation function.
  BuildDropConnectNetwork<>
      (dataset, labels, dataset, labels, 2, 10, 50, 0.2);
}

BOOST_AUTO_TEST_SUITE_END();