/**
 * @file rmsprop_test.cpp
 * @author Marcus Edel
 *
 * Tests the RMSProp optimizer on a couple test models.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>

#include <mlpack/methods/ann/trainer/trainer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(RMSPropTest);

/**
 * Train and evaluate a vanilla network with the specified structure. Using the
 * iris data, the data set contains 3 classes. One class is linearly separable
 * from the other 2. The other two aren't linearly separable from each other.
 */
BOOST_AUTO_TEST_CASE(SimpleRMSPropTestFunction)
{
  const size_t hiddenLayerSize = 10;
  const size_t maxEpochs = 300;

  // Load the dataset.
  arma::mat dataset, labels, labelsIdx;
  data::Load("iris_train.csv", dataset, true);
  data::Load("iris_train_labels.csv", labelsIdx, true);

  // Create target matrix.
  labels = arma::zeros<arma::mat>(labelsIdx.max() + 1, labelsIdx.n_cols);
  for (size_t i = 0; i < labelsIdx.n_cols; i++)
    labels(labelsIdx(0, i), i) = 1;

  // Construct a feed forward network using the specified parameters.
  RandomInitialization randInit(0.1, 0.1);

  LinearLayer<RMSPROP, RandomInitialization> inputLayer(dataset.n_rows,
      hiddenLayerSize, randInit);
  BiasLayer<RMSPROP, RandomInitialization> inputBiasLayer(hiddenLayerSize,
      1, randInit);
  BaseLayer<LogisticFunction> inputBaseLayer;

  LinearLayer<RMSPROP, RandomInitialization> hiddenLayer1(hiddenLayerSize,
      labels.n_rows, randInit);
  BiasLayer<RMSPROP, RandomInitialization> hiddenBiasLayer1(labels.n_rows,
      1, randInit);
  BaseLayer<LogisticFunction> outputLayer;

  OneHotLayer classOutputLayer;

  auto modules = std::tie(inputLayer, inputBiasLayer, inputBaseLayer,
                          hiddenLayer1, hiddenBiasLayer1, outputLayer);

  FFN<decltype(modules), OneHotLayer, MeanSquaredErrorFunction>
      net(modules, classOutputLayer);

  arma::mat prediction;
  size_t error = 0;

  // Evaluate the feed forward network.
  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    arma::mat input = dataset.unsafe_col(i);
    net.Predict(input, prediction);

    if (arma::sum(arma::sum(arma::abs(
      prediction - labels.unsafe_col(i)))) == 0)
      error++;
  }

  // Check if the selected model isn't already optimized.
  double classificationError = 1 - double(error) / dataset.n_cols;
  BOOST_REQUIRE_GE(classificationError, 0.09);

  // Train the feed forward network.
  Trainer<decltype(net)> trainer(net, maxEpochs, 1, 0.01, false);
  trainer.Train(dataset, labels, dataset, labels);

  // Evaluate the feed forward network.
  error = 0;
  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    arma::mat input = dataset.unsafe_col(i);
    net.Predict(input, prediction);

    if (arma::sum(arma::sum(arma::abs(
      prediction - labels.unsafe_col(i)))) == 0)
      error++;
  }

  classificationError = 1 - double(error) / dataset.n_cols;

  BOOST_REQUIRE_LE(classificationError, 0.09);
}

BOOST_AUTO_TEST_SUITE_END();
