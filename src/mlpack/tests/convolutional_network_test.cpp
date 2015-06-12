/**
 * @file convolutional_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the convolutional neural network.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <mlpack/methods/ann/connections/full_connection.hpp>
#include <mlpack/methods/ann/connections/bias_connection.hpp>
#include <mlpack/methods/ann/connections/conv_connection.hpp>
#include <mlpack/methods/ann/connections/pooling_connection.hpp>

#include <mlpack/methods/ann/layer/neuron_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>

#include <mlpack/methods/ann/cnn.hpp>
#include <mlpack/methods/ann/trainer/trainer.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>

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
      Y.col(i)(0) = 1;
    }
    else
    {
      Y.col(i)(1) = 1;
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

  NeuronLayer<LogisticFunction, arma::cube> inputLayer(28, 28, 1);

  ConvLayer<LogisticFunction> convLayer0(24, 24, inputLayer.LayerSlices(), 6);
  ConvConnection<decltype(inputLayer),
                 decltype(convLayer0)>
      con1(inputLayer, convLayer0, 5);

  BiasLayer<> biasLayer0(6);
  BiasConnection<decltype(biasLayer0),
                 decltype(convLayer0)>
  con1Bias(biasLayer0, convLayer0);

  con1.Weights().slice(0) = arma::mat(
      "-0.0307   -0.1510   -0.0299    0.0631    0.1114;"
      "0.0816   -0.1162    0.0686   -0.0306    0.1734;"
      "-0.1851   -0.0572   -0.1094    0.0217   -0.0691;"
      "-0.0732   -0.0382    0.1400   -0.1332    0.0712;"
      "-0.1308    0.0144   -0.1750   -0.1118    0.1394");

  con1.Weights().slice(1) = arma::mat(
      "0.1461   -0.1487   -0.0683    0.1810   -0.0193;"
      "-0.1537   -0.0292    0.0691    0.0919    0.1513;"
      "-0.1707    0.1696    0.1239   -0.0813   -0.0764;"
      "-0.1223    0.0123   -0.1784    0.1071   -0.0786;"
      "0.1400    0.0711    0.0926   -0.1469   -0.1370;");

  con1.Weights().slice(2) = arma::mat(
      "-0.1780   -0.1654   -0.1473    0.0133    0.1494;"
      "0.0662    0.0274   -0.0318    0.0607   -0.1343;"
      "-0.1068   -0.1308    0.0720    0.0055   -0.1336;"
      "-0.0868    0.0331   -0.0318    0.1646    0.1138;"
      "-0.0031    0.0740   -0.1667    0.0321   -0.0379;");

  con1.Weights().slice(3) = arma::mat(
      "-0.1239    0.1419    0.1466   -0.1427   -0.0974;"
      "0.1583    0.0458   -0.0266    0.1665    0.1494;"
      "-0.0564    0.0929    0.1721   -0.0185    0.0273;"
      "0.0929   -0.0560    0.0605    0.0290   -0.1841;"
      "0.0837   -0.0852    0.0451   -0.0340    0.0434;");

  con1.Weights().slice(4) = arma::mat(
      "-0.0642    0.0457   -0.1213    0.0946   -0.1778;"
      "0.0100   -0.1793   -0.1344    0.0940   -0.1755;"
      "0.1429    0.1590    0.1602    0.1567   -0.1747;"
      "-0.0529    0.0707    0.0729    0.0783   -0.0940;"
      "0.1513    0.1842   -0.1607   -0.1391    0.1333;");

  con1.Weights().slice(5) = arma::mat(
      "0.0144    0.0318   -0.0989    0.0208   -0.1454;"
      "0.0196    0.1739    0.1137   -0.1346   -0.1016;"
      "0.1267    0.0226   -0.0415   -0.1630    0.0789;"
      "-0.1392   -0.1783    0.1346   -0.1402    0.0221;"
      "-0.0818    0.1113    0.0915   -0.1687   -0.1805;");


  PoolingLayer<> poolingLayer0(12, 12, inputLayer.LayerSlices(), 6);
  PoolingConnection<decltype(convLayer0),
                    decltype(poolingLayer0)>
      con2(convLayer0, poolingLayer0);

  ConvLayer<LogisticFunction> convLayer1(8, 8, inputLayer.LayerSlices(), 12);
  ConvConnection<decltype(poolingLayer0),
                 decltype(convLayer1)>
      con3(poolingLayer0, convLayer1, 5);

  BiasLayer<> biasLayer3(12);
  BiasConnection<decltype(biasLayer3),
                 decltype(convLayer1)>
  con3Bias(biasLayer3, convLayer1);

  PoolingLayer<> poolingLayer1(4, 4, inputLayer.LayerSlices(), 12);
  PoolingConnection<decltype(convLayer1),
                    decltype(poolingLayer1)>
      con4(convLayer1, poolingLayer1);

  NeuronLayer<LogisticFunction, arma::mat> outputLayer(10,
      inputLayer.LayerSlices());
  FullConnection<decltype(poolingLayer1),
                 decltype(outputLayer)>
    con5(poolingLayer1, outputLayer);

  BiasLayer<> biasLayer1(1);
  FullConnection<decltype(biasLayer1),
                 decltype(outputLayer)>
    con5Bias(biasLayer1, outputLayer);


  BinaryClassificationLayer finalOutputLayer;

  auto module0 = std::tie(con1, con1Bias);
  auto module1 = std::tie(con2);
  auto module2 = std::tie(con3, con3Bias);
  auto module3 = std::tie(con4);
  auto module4 = std::tie(con5, con5Bias);
  auto modules = std::tie(module0, module1, module2, module3, module4);

  CNN<decltype(modules), decltype(finalOutputLayer),
      MeanSquaredErrorFunction> net(modules, finalOutputLayer);

  Trainer<decltype(net)> trainer(net, 1);

  for (size_t j = 0; j < 40; ++j)
  {
    arma::Col<size_t> index = arma::linspace<arma::Col<size_t> >(200,
        299, 300);
    index = arma::shuffle(index);

    for (size_t i = 200; i < 300; i++)
    {

      arma::cube input = arma::cube(28, 28, 1);
      input.slice(0) = arma::mat(X.colptr(index(i)), 28, 28);

      arma::mat labels = arma::mat(10, 1);
      labels.col(0) = Y.col(index(i));

      trainer.Train(input, labels, input, labels);
    }
  }

  size_t error = 0;
  for (size_t i = 200; i < 300; i++)
  {
    arma::cube input = arma::cube(X.colptr(i), 28, 28, 1);
    arma::mat labels = Y.col(i);

    arma::mat prediction;
    net.Predict(input, prediction);

    bool b = arma::all(arma::abs(
        arma::vectorise(prediction) - arma::vectorise(labels)) < 0.1);

    if (!b)
      error++;
  }

  BOOST_REQUIRE_LE(error, 30);
}

BOOST_AUTO_TEST_SUITE_END();
