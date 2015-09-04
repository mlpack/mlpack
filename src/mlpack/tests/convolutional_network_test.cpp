/**
 * @file convolutional_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the convolutional neural network.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/dropout_layer.hpp>

#include <mlpack/methods/ann/cnn.hpp>
#include <mlpack/methods/ann/trainer/trainer.hpp>
#include <mlpack/methods/ann/optimizer/ada_delta.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;


BOOST_AUTO_TEST_SUITE(ConvolutionalNetworkTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<
    typename PerformanceFunction
>
void BuildVanillaNetwork()
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
      Y.col(i)(5) = 1;
    }
    else
    {
      Y.col(i)(8) = 1;
    }
  }

  arma::cube input = arma::cube(28, 28, nPoints);
  for (size_t i = 0; i < nPoints; i++)
    input.slice(i) = arma::mat(X.colptr(i), 28, 28);

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

  ConvLayer<RMSPROP> convLayer0(1, 8, 5, 5);
  BiasLayer2D<RMSPROP, ZeroInitialization> biasLayer0(8);
  BaseLayer2D<PerformanceFunction> baseLayer0;
  PoolingLayer<> poolingLayer0(2);

  ConvLayer<RMSPROP> convLayer1(8, 12, 5, 5);
  BiasLayer2D<RMSPROP, ZeroInitialization> biasLayer1(12);
  BaseLayer2D<PerformanceFunction> baseLayer1;
  PoolingLayer<> poolingLayer1(2);

  LinearMappingLayer<RMSPROP> linearLayer0(192, 10);
  BiasLayer<RMSPROP> biasLayer2(10);
  SoftmaxLayer<> softmaxLayer0;

  OneHotLayer outputLayer;

  auto modules = std::tie(convLayer0, biasLayer0, baseLayer0, poolingLayer0,
                          convLayer1, biasLayer1, baseLayer1, poolingLayer1,
                          linearLayer0, biasLayer2, softmaxLayer0);

  CNN<decltype(modules), decltype(outputLayer)>
      net(modules, outputLayer);

  Trainer<decltype(net)> trainer(net, 50, 1, 0.7);
  trainer.Train(input, Y, input, Y);

  BOOST_REQUIRE_LE(trainer.ValidationError(), 0.7);
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  BuildVanillaNetwork<LogisticFunction>();
}

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<
    typename PerformanceFunction
>
void BuildVanillaDropoutNetwork()
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
      Y.col(i)(5) = 1;
    }
    else
    {
      Y.col(i)(8) = 1;
    }
  }

  arma::cube input = arma::cube(28, 28, nPoints);
  for (size_t i = 0; i < nPoints; i++)
    input.slice(i) = arma::mat(X.colptr(i), 28, 28);

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

  ConvLayer<AdaDelta> convLayer0(1, 4, 5, 5);
  BiasLayer2D<AdaDelta, ZeroInitialization> biasLayer0(4);
  DropoutLayer2D<> dropoutLayer0;
  BaseLayer2D<PerformanceFunction> baseLayer0;
  PoolingLayer<> poolingLayer0(2);

  ConvLayer<AdaDelta> convLayer1(4, 8, 5, 5);
  BiasLayer2D<AdaDelta, ZeroInitialization> biasLayer1(8);
  BaseLayer2D<PerformanceFunction> baseLayer1;
  PoolingLayer<> poolingLayer1(2);

  LinearMappingLayer<AdaDelta> linearLayer0(128, 10);
  BiasLayer<AdaDelta> biasLayer2(10);
  SoftmaxLayer<> softmaxLayer0;

  OneHotLayer outputLayer;

  auto modules = std::tie(convLayer0, biasLayer0, dropoutLayer0, baseLayer0,
                          poolingLayer0, convLayer1, biasLayer1, baseLayer1,
                          poolingLayer1, linearLayer0, biasLayer2,
                          softmaxLayer0);

  CNN<decltype(modules), decltype(outputLayer)>
      net(modules, outputLayer);

  Trainer<decltype(net)> trainer(net, 50, 1, 0.7);
  trainer.Train(input, Y, input, Y);

  BOOST_REQUIRE_LE(trainer.ValidationError(), 0.7);
}

/**
 * Train the network on a larger dataset using dropout.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkDropoutTest)
{
  BuildVanillaDropoutNetwork<RectifierFunction>();
}

BOOST_AUTO_TEST_SUITE_END();
