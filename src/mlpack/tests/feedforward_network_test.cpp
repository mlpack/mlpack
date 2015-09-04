/**
 * @file feedforward_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the feed forward network.
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

#include <mlpack/methods/ann/trainer/trainer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;


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
                         const double classificationErrorThreshold,
                         const double ValidationErrorThreshold)
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

  FFN<decltype(modules), decltype(classOutputLayer), PerformanceFunctionType>
      net(modules, classOutputLayer);

  Trainer<decltype(net)> trainer(net, maxEpochs, 1, 0.01);
  trainer.Train(trainData, trainLabels, testData, testLabels);

  MatType prediction;
  size_t error = 0;

  for (size_t i = 0; i < testData.n_cols; i++)
  {
  	MatType predictionInput = testData.unsafe_col(i);
  	MatType targetOutput = testLabels.unsafe_col(i);

    net.Predict(predictionInput, prediction);

    if (arma::sum(arma::sum(arma::abs(prediction - targetOutput))) == 0)
      error++;
  }

  double classificationError = 1 - double(error) / testData.n_cols;

  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
  BOOST_REQUIRE_LE(trainer.ValidationError(), ValidationErrorThreshold);
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
      (trainData, trainLabels, testData, testLabels, 4, 500, 0.1, 60);

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
      (dataset, labels, dataset, labels, 30, 100, 0.6, 10);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
    (dataset, labels, dataset, labels, 10, 200, 0.6, 20);
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
                         const double classificationErrorThreshold,
                         const double ValidationErrorThreshold)
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

  FFN<decltype(modules), decltype(classOutputLayer), PerformanceFunctionType>
      net(modules, classOutputLayer);

  Trainer<decltype(net)> trainer(net, maxEpochs, 1, 0.001);
  trainer.Train(trainData, trainLabels, testData, testLabels);

  MatType prediction;
  size_t error = 0;

  for (size_t i = 0; i < testData.n_cols; i++)
  {
  	MatType input = testData.unsafe_col(i);
    net.Predict(input, prediction);
    if (arma::sum(arma::sum(arma::abs(
    	prediction - testLabels.unsafe_col(i)))) == 0)
      error++;
  }

  double classificationError = 1 - double(error) / testData.n_cols;

  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
  BOOST_REQUIRE_LE(trainer.ValidationError(), ValidationErrorThreshold);
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
      (trainData, trainLabels, testData, testLabels, 4, 100, 0.1, 60);

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
      (dataset, labels, dataset, labels, 8, 100, 0.6, 10);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
    (dataset, labels, dataset, labels, 8, 100, 0.6, 20);
}

/**
 * Train the network until the validation error converge.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkConvergenceTest)
{
  arma::mat input;
  arma::mat labels;

  // Test on a non-linearly separable dataset (XOR).
  input << 0 << 1 << 1 << 0 << arma::endr
        << 1 << 0 << 1 << 0 << arma::endr;
  labels << 0 << 0 << 1 << 1;

  // Vanilla neural net with logistic activation function.
  BuildVanillaNetwork<LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (input, labels, input, labels, 4, 5000, 0, 0.01);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (input, labels, input, labels, 4, 5000, 0, 0.01);

  // Test on a linearly separable dataset (AND).
  input << 0 << 1 << 1 << 0 << arma::endr
        << 1 << 0 << 1 << 0 << arma::endr;
  labels << 0 << 0 << 1 << 0;

  // vanilla neural net with sigmoid activation function.
  BuildVanillaNetwork<LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
    (input, labels, input, labels, 4, 5000, 0, 0.01);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (input, labels, input, labels, 4, 5000, 0, 0.01);
}

/**
 * Train a vanilla network with the specified structure step by step and
 * evaluate the network.
 */
template<
    typename PerformanceFunction,
    typename OutputLayerType,
    typename PerformanceFunctionType,
    typename MatType = arma::mat
>
void BuildNetworkOptimzer(MatType& trainData,
                          MatType& trainLabels,
                          MatType& testData,
                          MatType& testLabels,
                          size_t hiddenLayerSize,
                          size_t epochs)
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

  RandomInitialization randInit(0.5, 0.5);

  LinearLayer<RMSPROP, RandomInitialization> inputLayer(trainData.n_rows,
      hiddenLayerSize, randInit);
  BiasLayer<RMSPROP, RandomInitialization> inputBiasLayer(hiddenLayerSize,
      1, randInit);
  BaseLayer<PerformanceFunction> inputBaseLayer;

  LinearLayer<RMSPROP, RandomInitialization> hiddenLayer1(hiddenLayerSize,
      trainLabels.n_rows, randInit);
  BiasLayer<RMSPROP, RandomInitialization> hiddenBiasLayer1(trainLabels.n_rows,
      1, randInit);
  BaseLayer<PerformanceFunction> outputLayer;

  OutputLayerType classOutputLayer;

  auto modules = std::tie(inputLayer, inputBiasLayer, inputBaseLayer,
  						  hiddenLayer1, hiddenBiasLayer1, outputLayer);

  FFN<decltype(modules), OutputLayerType, PerformanceFunctionType>
      net(modules, classOutputLayer);

  Trainer<decltype(net)> trainer(net, epochs, 1, 0.0001, false);

  double error = DBL_MAX;
  for (size_t i = 0; i < 5; i++)
  {
    trainer.Train(trainData, trainLabels, testData, testLabels);
    double validationError = trainer.ValidationError();

    bool b = validationError < error || validationError == 0;
    BOOST_REQUIRE_EQUAL(b, 1);

    error = validationError;
  }
}

/**
 * Train the network with different optimzer and check if the error decreases
 * over time.
 */
BOOST_AUTO_TEST_CASE(NetworkDecreasingErrorTest)
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1) += 1;

  // Vanilla neural net with logistic activation function.
  BuildNetworkOptimzer<LogisticFunction,
                       BinaryClassificationLayer,
                       MeanSquaredErrorFunction>
      (dataset, labels, dataset, labels, 20, 15);
}

BOOST_AUTO_TEST_SUITE_END();
