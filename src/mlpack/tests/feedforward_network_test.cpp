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

#include <mlpack/methods/ann/layer/neuron_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>

#include <mlpack/methods/ann/connections/full_connection.hpp>

#include <mlpack/methods/ann/trainer/trainer.hpp>

#include <mlpack/methods/ann/ffnn.hpp>

#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/performance_functions/sse_function.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>

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
    typename MatType = arma::mat,
    typename VecType = arma::colvec
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
   * |     |     +>|     |       |     |
   * +-----+     | +--+--+       +-----+
   *             |
   *  Bias       |
   *  Layer      |
   * +-----+     |
   * |     |     |
   * |     +-----+
   * |     |
   * +-----+
   */
  BiasLayer<> biasLayer0(1);

  NeuronLayer<PerformanceFunction> inputLayer(trainData.n_rows);
  NeuronLayer<PerformanceFunction> hiddenLayer0(hiddenLayerSize);
  NeuronLayer<PerformanceFunction> hiddenLayer1(trainLabels.n_rows);

  OutputLayerType outputLayer;

  FullConnection<
    decltype(inputLayer),
    decltype(hiddenLayer0)>
    layerCon0(inputLayer, hiddenLayer0);

  FullConnection<
    decltype(biasLayer0),
    decltype(hiddenLayer0)>
    layerCon1(biasLayer0, hiddenLayer0);

  FullConnection<
      decltype(hiddenLayer0),
      decltype(hiddenLayer1)>
      layerCon2(hiddenLayer0, hiddenLayer1);

  auto module0 = std::tie(layerCon0, layerCon1);
  auto module1 = std::tie(layerCon2);
  auto modules = std::tie(module0, module1);

  FFNN<decltype(modules), decltype(outputLayer), PerformanceFunctionType>
      net(modules, outputLayer);

  Trainer<decltype(net)> trainer(net, maxEpochs, 1, 0.001);
  trainer.Train(trainData, trainLabels, testData, testLabels);

  VecType prediction;
  size_t error = 0;

  for (size_t i = 0; i < testData.n_cols; i++)
  {
    net.Predict(testData.unsafe_col(i), prediction);
    if (arma::sum(prediction - testLabels.unsafe_col(i)) == 0)
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
      (dataset, labels, dataset, labels, 100, 100, 0.6, 10);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                    BinaryClassificationLayer,
                    MeanSquaredErrorFunction>
    (dataset, labels, dataset, labels, 10, 200, 0.6, 20);
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
      (input, labels, input, labels, 4, 0, 0, 0.01);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (input, labels, input, labels, 4, 0, 0, 0.01);

  // Test on a linearly separable dataset (AND).
  input << 0 << 1 << 1 << 0 << arma::endr
        << 1 << 0 << 1 << 0 << arma::endr;
  labels << 0 << 0 << 1 << 0;

  // vanilla neural net with sigmoid activation function.
  BuildVanillaNetwork<LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
    (input, labels, input, labels, 4, 0, 0, 0.01);

  // Vanilla neural net with tanh activation function.
  BuildVanillaNetwork<TanhFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (input, labels, input, labels, 4, 0, 0, 0.01);
}

/**
 * Train a vanilla network with the specified structure step by step and
 * evaluate the network.
 */
template<
    typename WeightInitRule,
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
                          size_t epochs,
                          WeightInitRule weightInitRule = WeightInitRule())
{
   /* Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Output
   *  Layer         Layer         Layer
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |     +>|     |       |     |
   * +-----+     | +--+--+       +-----+
   *             |
   *  Bias       |
   *  Layer      |
   * +-----+     |
   * |     |     |
   * |     +-----+
   * |     |
   * +-----+
   */
  BiasLayer<> biasLayer0(1);

  NeuronLayer<PerformanceFunction> inputLayer(trainData.n_rows);
  NeuronLayer<PerformanceFunction> hiddenLayer0(hiddenLayerSize);
  NeuronLayer<PerformanceFunction> hiddenLayer1(trainLabels.n_rows);

  OutputLayerType outputLayer;

  FullConnection<
    decltype(inputLayer),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(weightInitRule)>
    layerCon0(inputLayer, hiddenLayer0, weightInitRule);

  FullConnection<
    decltype(biasLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(weightInitRule)>
    layerCon1(biasLayer0, hiddenLayer0, weightInitRule);

  FullConnection<
      decltype(hiddenLayer0),
      decltype(hiddenLayer1),
      mlpack::ann::RMSPROP,
      decltype(weightInitRule)>
      layerCon2(hiddenLayer0, hiddenLayer1, weightInitRule);

  auto module0 = std::tie(layerCon0, layerCon1);
  auto module1 = std::tie(layerCon2);
  auto modules = std::tie(module0, module1);

  FFNN<decltype(modules), decltype(outputLayer), PerformanceFunctionType>
      net(modules, outputLayer);

  Trainer<decltype(net)> trainer(net, epochs, 1);

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

  RandomInitialization randInitB(-0.5, 0.5);

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1) += 1;

  // Vanilla neural net with logistic activation function.
  BuildNetworkOptimzer<RandomInitialization,
                       LogisticFunction,
                       BinaryClassificationLayer,
                       MeanSquaredErrorFunction>
      (dataset, labels, dataset, labels, 100, 50, randInitB);
}

BOOST_AUTO_TEST_SUITE_END();
