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

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/dropout_layer.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>
#include <mlpack/methods/ann/layer/dropconnect_layer.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(FeedForwardNetworkTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<
    typename PerformanceFunction,
    typename OutputLayerType,
    typename PerformanceFunctionType,
    typename MatType = arma::mat
>
void BuildVanillaNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
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

  LinearLayer<> inputLayer(trainData.n_rows, hiddenLayerSize);
  BiasLayer<> inputBiasLayer(hiddenLayerSize);
  BaseLayer<PerformanceFunction> inputBaseLayer;

  LinearLayer<> hiddenLayer1(hiddenLayerSize, trainLabels.n_rows);
  BiasLayer<> hiddenBiasLayer1(trainLabels.n_rows);
  BaseLayer<PerformanceFunction> outputLayer;

  OutputLayerType classOutputLayer;

  auto modules = std::tie(inputLayer, inputBiasLayer, inputBaseLayer,
                          hiddenLayer1, hiddenBiasLayer1, outputLayer);

  FFN<decltype(modules), decltype(classOutputLayer), RandomInitialization,
      PerformanceFunctionType> net(modules, classOutputLayer);

  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, 1e-18);

  net.Train(trainData, trainLabels, opt);

  MatType prediction;
  net.Predict(testData, prediction);

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (arma::sum(arma::sum(
        arma::abs(prediction.col(i) - testLabels.col(i)))) == 0)
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
  arma::mat trainLabels = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);
  arma::mat testLabels = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildVanillaNetwork<LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (trainData, trainLabels, testData, testLabels, 8, 200, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  // Vanilla neural net with logistic activation function.
  BuildVanillaNetwork<LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (dataset, labels, dataset, labels, 30, 30, 0.4);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
    (dataset, labels, dataset, labels, 10, 30, 0.4);
}

/**
 * Train and evaluate a Dropout network with the specified structure.
 */
template<
    typename PerformanceFunction,
    typename OutputLayerType,
    typename PerformanceFunctionType,
    typename MatType = arma::mat
>
void BuildDropoutNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
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

  LinearLayer<> inputLayer(trainData.n_rows, hiddenLayerSize);
  BiasLayer<> biasLayer(hiddenLayerSize);
  BaseLayer<PerformanceFunction> hiddenLayer0;
  DropoutLayer<> dropoutLayer0;

  LinearLayer<> hiddenLayer1(hiddenLayerSize, trainLabels.n_rows);
  BaseLayer<PerformanceFunction> outputLayer;

  OutputLayerType classOutputLayer;

  auto modules = std::tie(inputLayer, biasLayer, hiddenLayer0, dropoutLayer0,
                          hiddenLayer1, outputLayer);

  FFN<decltype(modules), decltype(classOutputLayer), RandomInitialization,
      PerformanceFunctionType> net(modules, classOutputLayer);

  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, 1e-18);

  net.Train(trainData, trainLabels, opt);

  MatType prediction;
  net.Predict(testData, prediction);

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (arma::sum(arma::sum(
        arma::abs(prediction.col(i) - testLabels.col(i)))) == 0)
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
  arma::mat trainLabels = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);
  arma::mat testLabels = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildDropoutNetwork<LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (trainData, trainLabels, testData, testLabels, 4, 100, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  // Vanilla neural net with logistic activation function.
  BuildDropoutNetwork<LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (dataset, labels, dataset, labels, 8, 30, 0.4);

  // Vanilla neural net with tanh activation function.
  BuildDropoutNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
    (dataset, labels, dataset, labels, 8, 30, 0.4);
}

/**
 * Train and evaluate a DropConnect network(with a baselayer) with the
 * specified structure.
 */
template<
    typename PerformanceFunction,
    typename OutputLayerType,
    typename PerformanceFunctionType,
    typename MatType = arma::mat
>
void BuildDropConnectNetwork(MatType& trainData,
                             MatType& trainLabels,
                             MatType& testData,
                             MatType& testLabels,
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
  LinearLayer<> inputLayer(trainData.n_rows, hiddenLayerSize);
  BiasLayer<> biasLayer(hiddenLayerSize);
  BaseLayer<PerformanceFunction> hiddenLayer0;

  LinearLayer<> hiddenLayer1(hiddenLayerSize, trainLabels.n_rows);
  DropConnectLayer<decltype(hiddenLayer1)> dropConnectLayer0(hiddenLayer1);

  BaseLayer<PerformanceFunction> outputLayer;

  OutputLayerType classOutputLayer;

  auto modules = std::tie(inputLayer, biasLayer, hiddenLayer0,
                          dropConnectLayer0, outputLayer);

  FFN<decltype(modules), decltype(classOutputLayer), RandomInitialization,
              PerformanceFunctionType> net(modules, classOutputLayer);

  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, 1e-18);

  net.Train(trainData, trainLabels, opt);

  MatType prediction;
  net.Predict(testData, prediction);

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
      if (arma::sum(arma::sum(
          arma::abs(prediction.col(i) - testLabels.col(i)))) == 0)
      {
          error++;
      }
  }

  double classificationError = 1 - double(error) / testData.n_cols;
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train and evaluate a DropConnect network(with a linearlayer) with the
 * specified structure.
 */
template<
    typename PerformanceFunction,
    typename OutputLayerType,
    typename PerformanceFunctionType,
    typename MatType = arma::mat
>
void BuildDropConnectNetworkLinear(MatType& trainData,
                                   MatType& trainLabels,
                                   MatType& testData,
                                   MatType& testLabels,
                                   const size_t hiddenLayerSize,
                                   const size_t maxEpochs,
                                   const double classificationErrorThreshold)
{
 /*
  * Construct a feed forward network with trainData.n_rows input nodes,
  * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
  * network struct that looks like:
  *
  * Input         Hidden       DropConnect     Output
  * Layer         Layer          Layer         Layer
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
  LinearLayer<> inputLayer(trainData.n_rows, hiddenLayerSize);
  BiasLayer<> biasLayer(hiddenLayerSize);
  BaseLayer<PerformanceFunction> hiddenLayer0;

  DropConnectLayer<> dropConnectLayer0(hiddenLayerSize, trainLabels.n_rows);

  BaseLayer<PerformanceFunction> outputLayer;

  OutputLayerType classOutputLayer;
  auto modules = std::tie(inputLayer, biasLayer, hiddenLayer0,
                          dropConnectLayer0, outputLayer);

  FFN<decltype(modules), decltype(classOutputLayer), RandomInitialization,
              PerformanceFunctionType> net(modules, classOutputLayer);

  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, 1e-18);

  net.Train(trainData, trainLabels, opt);

  MatType prediction;
  net.Predict(testData, prediction);

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
      if (arma::sum(arma::sum(
          arma::abs(prediction.col(i) - testLabels.col(i)))) == 0)
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
  arma::mat trainLabels = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);
  arma::mat testLabels = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildDropConnectNetwork<LogisticFunction,
                          BinaryClassificationLayer,
                          MeanSquaredErrorFunction>
      (trainData, trainLabels, testData, testLabels, 4, 100, 0.1);

  BuildDropConnectNetworkLinear<LogisticFunction,
                                BinaryClassificationLayer,
                                MeanSquaredErrorFunction>
      (trainData, trainLabels, testData, testLabels, 4, 100, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  // Vanilla neural net with logistic activation function.
  BuildDropConnectNetwork<LogisticFunction,
                          BinaryClassificationLayer,
                          MeanSquaredErrorFunction>
      (dataset, labels, dataset, labels, 8, 30, 0.4);


  BuildDropConnectNetworkLinear<LogisticFunction,
                                BinaryClassificationLayer,
                                MeanSquaredErrorFunction>
      (dataset, labels, dataset, labels, 8, 30, 0.4);
}

BOOST_AUTO_TEST_SUITE_END();
