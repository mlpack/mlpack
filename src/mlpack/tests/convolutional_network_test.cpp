/**
 * @file convolutional_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the convolutional neural network.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

#include <mlpack/methods/ann/connections/full_connection.hpp>
#include <mlpack/methods/ann/connections/identity_connection.hpp>
#include <mlpack/methods/ann/connections/bias_connection.hpp>
#include <mlpack/methods/ann/connections/conv_connection.hpp>
#include <mlpack/methods/ann/connections/pooling_connection.hpp>

#include <mlpack/methods/ann/layer/neuron_layer.hpp>
#include <mlpack/methods/ann/layer/dropout_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>

#include <mlpack/methods/ann/cnn.hpp>
#include <mlpack/methods/ann/trainer/trainer.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/optimizer/ada_delta.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;


BOOST_AUTO_TEST_SUITE(ConvolutionalNetworkTest);

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  arma::mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  arma::uword nPoints = X.n_cols;
  for (arma::uword i = 0; i < nPoints; i++)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  // Build the target matrix.
  arma::mat Y = arma::zeros<arma::mat>(10, nPoints);
  for (size_t i = 0; i < nPoints; i++)
  {
    if (i < nPoints / 2)
    {
      Y.col(i)(1) = 1;
    }
    else
    {
      Y.col(i)(8) = 1;
    }
  }

  /*
   * Construct a convolutional neural network with a 28x28x1 input layer,
   * 24x24x6 convolution layer, 12x12x6 pooling layer, 8x8x12 convolution layer
   * and a 4x4x12 pooling layer which is fully connected with the output layer.
   * The network structure looks like:
   *
   * Input    Convolution  Pooling      Convolution  Pooling      Output
   * Layer    Layer        Layer        Layer        Layer        Layer
   *
   *          +---+        +---+        +---+        +---+
   *          | +---+      | +---+      | +---+      | +---+
   * +---+    | | +---+    | | +---+    | | +---+    | | +---+    +---+
   * |   |    | | |   |    | | |   |    | | |   |    | | |   |    |   |
   * |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> |   |
   * |   |      +-+   |      +-+   |      +-+   |      +-+   |    |   |
   * +---+        +---+        +---+        +---+        +---+    +---+
   */

  NeuronLayer<RectifierFunction, arma::cube> inputLayer(28, 28, 1);

  ConvLayer<RectifierFunction> convLayer0(24, 24, inputLayer.LayerSlices(), 6);
  ConvConnection<decltype(inputLayer),
                 decltype(convLayer0),
                 mlpack::ann::AdaDelta>
      con1(inputLayer, convLayer0, 5);

  BiasLayer<> biasLayer0(6);
  BiasConnection<decltype(biasLayer0),
                 decltype(convLayer0),
                 mlpack::ann::AdaDelta,
                 mlpack::ann::ZeroInitialization>
  con1Bias(biasLayer0, convLayer0);

  PoolingLayer<> poolingLayer0(12, 12, inputLayer.LayerSlices(), 6);
  PoolingConnection<decltype(convLayer0),
                    decltype(poolingLayer0)>
      con2(convLayer0, poolingLayer0);

  ConvLayer<RectifierFunction> convLayer1(8, 8, inputLayer.LayerSlices(), 12);
  ConvConnection<decltype(poolingLayer0),
                 decltype(convLayer1),
                 mlpack::ann::AdaDelta>
      con3(poolingLayer0, convLayer1, 5);

  BiasLayer<> biasLayer3(12);
  BiasConnection<decltype(biasLayer3),
                 decltype(convLayer1),
                 mlpack::ann::AdaDelta,
                 mlpack::ann::ZeroInitialization>
  con3Bias(biasLayer3, convLayer1);

  PoolingLayer<> poolingLayer1(4, 4, inputLayer.LayerSlices(), 12);
  PoolingConnection<decltype(convLayer1),
                    decltype(poolingLayer1)>
      con4(convLayer1, poolingLayer1);

  SoftmaxLayer<arma::mat> outputLayer(10,
    inputLayer.LayerSlices());

  FullConnection<decltype(poolingLayer1),
                 decltype(outputLayer),
                 mlpack::ann::AdaDelta>
    con5(poolingLayer1, outputLayer);

  BiasLayer<> biasLayer1(1);
  FullConnection<decltype(biasLayer1),
                 decltype(outputLayer),
                 mlpack::ann::AdaDelta,
                 mlpack::ann::ZeroInitialization>
    con5Bias(biasLayer1, outputLayer);

  OneHotLayer finalOutputLayer;

  auto module0 = std::tie(con1, con1Bias);
  auto module1 = std::tie(con2);
  auto module2 = std::tie(con3, con3Bias);
  auto module3 = std::tie(con4);
  auto module4 = std::tie(con5, con5Bias);
  auto modules = std::tie(module0, module1, module2, module3, module4);

  CNN<decltype(modules), decltype(finalOutputLayer),
      MeanSquaredErrorFunction> net(modules, finalOutputLayer);

  Trainer<decltype(net)> trainer(net, 1);

  size_t error = 0;
  for (size_t j = 0; j < 30; ++j)
  {
    arma::Col<size_t> index = arma::linspace<arma::Col<size_t> >(0,
        499, 500);
    index = arma::shuffle(index);

    for (size_t i = 0; i < 500; i++)
    {
      arma::cube input = arma::cube(28, 28, 1);
      input.slice(0) = arma::mat(X.colptr(index(i)), 28, 28);

      arma::mat labels = arma::mat(10, 1);
      labels.col(0) = Y.col(index(i));

      trainer.Train(input, labels, input, labels);
    }

    error = 0;
    for (size_t p = 0; p < 500; p++)
    {
      arma::cube input = arma::cube(X.colptr(p), 28, 28, 1);
      arma::mat labels = Y.col(p);

      arma::mat prediction;
      net.Predict(input, prediction);

      bool b = arma::all(arma::abs(
          arma::vectorise(prediction) - arma::vectorise(labels)) < 0.1);

      if (!b)
        error++;
    }

    if (error <= 10)
      break;
  }

  BOOST_REQUIRE_LE(error, 10);
}

/**
 * Train the network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkDropoutTest)
{
  arma::mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  arma::uword nPoints = X.n_cols;
  for (arma::uword i = 0; i < nPoints; i++)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  // Build the target matrix.
  arma::mat Y = arma::zeros<arma::mat>(10, nPoints);
  for (size_t i = 0; i < nPoints; i++)
  {
    if (i < nPoints / 2)
    {
      Y.col(i)(1) = 1;
    }
    else
    {
      Y.col(i)(8) = 1;
    }
  }

  /*
   * Construct a convolutional neural network with a 28x28x1 input layer,
   * 24x24x6 convolution layer, 12x12x6 pooling layer, 8x8x12 convolution layer,
   * 8x8x12 Dropout Layer and a 4x4x12 pooling layer which is fully connected
   * with the output layer. The network structure looks like:
   *
   * Input    Convolution  Dropout      Pooling     Convolution,     Output
   * Layer    Layer        Layer        Layer       Dropout,         Layer
   *                                                Pooling Layer
   *          +---+        +---+        +---+
   *          | +---+      | +---+      | +---+
   * +---+    | | +---+    | | +---+    | | +---+                    +---+
   * |   |    | | |   |    | | |   |    | | |   |                    |   |
   * |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> ............--> |   |
   * |   |      +-+   |      +-+   |      +-+   |                    |   |
   * +---+        +---+        +---+        +---+                    +---+
   */

  NeuronLayer<RectifierFunction, arma::cube> inputLayer(28, 28, 1);

  ConvLayer<RectifierFunction> convLayer0(24, 24, inputLayer.LayerSlices(), 6);
  ConvConnection<decltype(inputLayer),
                 decltype(convLayer0),
                 mlpack::ann::AdaDelta>
      con1(inputLayer, convLayer0, 5);

  BiasLayer<> biasLayer0(6);
  BiasConnection<decltype(biasLayer0),
                 decltype(convLayer0),
                 mlpack::ann::AdaDelta,
                 mlpack::ann::ZeroInitialization>
      con1Bias(biasLayer0, convLayer0);

  DropoutLayer<arma::cube> dropoutLayer0(24, 24, inputLayer.LayerSlices(), 6);
  IdentityConnection<decltype(convLayer0),
                     decltype(dropoutLayer0)>
      con1Dropout(convLayer0, dropoutLayer0);

  PoolingLayer<> poolingLayer0(12, 12, inputLayer.LayerSlices(), 6);
  PoolingConnection<decltype(dropoutLayer0),
                  decltype(poolingLayer0)>
      con2(dropoutLayer0, poolingLayer0);

  ConvLayer<RectifierFunction> convLayer1(8, 8, inputLayer.LayerSlices(), 12);
  ConvConnection<decltype(poolingLayer0),
                 decltype(convLayer1),
                 mlpack::ann::AdaDelta>
      con3(poolingLayer0, convLayer1, 5);

  BiasLayer<> biasLayer3(12);
  BiasConnection<decltype(biasLayer3),
                 decltype(convLayer1),
                 mlpack::ann::AdaDelta,
                 mlpack::ann::ZeroInitialization>
      con3Bias(biasLayer3, convLayer1);

  DropoutLayer<arma::cube> dropoutLayer3(8, 8, inputLayer.LayerSlices(), 12);
  IdentityConnection<decltype(convLayer1),
                     decltype(dropoutLayer3)>
      con3Dropout(convLayer1, dropoutLayer3);

  PoolingLayer<> poolingLayer1(4, 4, inputLayer.LayerSlices(), 12);
  PoolingConnection<decltype(dropoutLayer3),
                    decltype(poolingLayer1)>
      con4(dropoutLayer3, poolingLayer1);

  SoftmaxLayer<arma::mat> outputLayer(10,
    inputLayer.LayerSlices());

  FullConnection<decltype(poolingLayer1),
                 decltype(outputLayer),
                 mlpack::ann::AdaDelta>
      con5(poolingLayer1, outputLayer);

  BiasLayer<> biasLayer1(1);
  FullConnection<decltype(biasLayer1),
                 decltype(outputLayer),
                 mlpack::ann::AdaDelta,
                 mlpack::ann::ZeroInitialization>
      con5Bias(biasLayer1, outputLayer);

  OneHotLayer finalOutputLayer;

  auto module0 = std::tie(con1, con1Bias);
  auto module0Dropout = std::tie(con1Dropout);
  auto module1 = std::tie(con2);
  auto module2 = std::tie(con3, con3Bias);
  auto module2Dropout = std::tie(con3Dropout);
  auto module3 = std::tie(con4);
  auto module4 = std::tie(con5, con5Bias);
  auto modules = std::tie(module0, module0Dropout, module1, module2,
      module2Dropout, module3, module4);

  CNN<decltype(modules), decltype(finalOutputLayer),
      MeanSquaredErrorFunction> net(modules, finalOutputLayer);

  Trainer<decltype(net)> trainer(net, 1);

  size_t error = 0;
  for (size_t j = 0; j < 30; ++j)
  {
    arma::Col<size_t> index = arma::linspace<arma::Col<size_t> >(0,
        499, 500);
    index = arma::shuffle(index);

    for (size_t i = 0; i < 500; i++)
    {
      arma::cube input = arma::cube(28, 28, 1);
      input.slice(0) = arma::mat(X.colptr(index(i)), 28, 28);

      arma::mat labels = arma::mat(10, 1);
      labels.col(0) = Y.col(index(i));

      trainer.Train(input, labels, input, labels);
    }

    error = 0;
    for (size_t p = 0; p < 500; p++)
    {
      arma::cube input = arma::cube(X.colptr(p), 28, 28, 1);
      arma::mat labels = Y.col(p);

      arma::mat prediction;
      net.Predict(input, prediction);

      bool b = arma::all(arma::abs(
          arma::vectorise(prediction) - arma::vectorise(labels)) < 0.1);

      if (!b)
        error++;
    }

    if (error <= 10)
      break;
  }

  BOOST_REQUIRE_LE(error, 10);
}

BOOST_AUTO_TEST_SUITE_END();
