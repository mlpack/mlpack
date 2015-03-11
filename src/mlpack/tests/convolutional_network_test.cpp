/**
 * @file convolutional_network_test.cpp
 * @author Shangtong Zhang
 *
 * Tests the convolutional network.
 */
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>

#include <mlpack/methods/ann/convolution/full_convolution.hpp>
#include <mlpack/methods/ann/convolution/valid_convolution.hpp>

#include <mlpack/methods/ann/connections/cnn_full_connection.hpp>
#include <mlpack/methods/ann/connections/cnn_bias_connection.hpp>
#include <mlpack/methods/ann/connections/cnn_conv_connection.hpp>
#include <mlpack/methods/ann/connections/cnn_pooling_connection.hpp>

#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/neuron_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>

#include <mlpack/methods/ann/pooling/max_pooling.hpp>
#include <mlpack/methods/ann/pooling/mean_pooling.hpp>

#include <mlpack/methods/ann/optimizer/steepest_descent.hpp>

#include <mlpack/methods/ann/cnn.hpp>

#include <mlpack/methods/ann/trainer/trainer.hpp>

#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;


BOOST_AUTO_TEST_SUITE(ConvolutionalNetworkTest);

// A simple test for convolution strategies.
BOOST_AUTO_TEST_CASE(ConvolutionTest)
{
  
  arma::mat convInput("1 2 3 4;"
                      "4 1 2 3;"
                      "3 4 1 2;"
                      "2 3 4 1;");
  
  arma::mat convKernel("1 0 -1;"
                       "0 1 0;"
                       "-1 0 1;");
  
  arma::mat correctValidConvOutput("-3 -2;"
                                   "8 -3;");
  
  arma::mat correctFullConvOutput("1 2 2 2 -3 -4;"
                                  "4 2 0 5 2 -3;"
                                  "2 6 -3 -2 5 2;"
                                  "-2 5 8 -3 0 2;"
                                  "-3 -2 5 6 2 2;"
                                  "-2 -3 -2 2 4 1;");
  
  arma::mat validConvOutput;
  
  ValidConvolution::conv(convInput, convKernel, validConvOutput);
  
  BOOST_CHECK(validConvOutput.n_elem ==
              sum(sum((validConvOutput == correctValidConvOutput))));
  
  arma::mat fullConvOutput;
  
  FullConvolution::conv(convInput, convKernel, fullConvOutput);
  
  BOOST_CHECK(fullConvOutput.n_elem ==
              sum(sum(fullConvOutput == correctFullConvOutput)));
  
}

// Helper function to build LeNet1, support for C++14 is required.
template <typename F, size_t... Is>
auto genTupleImpl(F func, std::index_sequence<Is...> )
{
  return std::make_tuple(func(Is)...);
}

template <size_t N, typename F>
auto genTuple(F func)
{
  return genTupleImpl(func, std::make_index_sequence<N>{} );
}

template <typename T, size_t I, typename... Tp1, typename... Tp2>
T genTupleImpl(std::tuple<Tp1...>& P1, std::tuple<Tp2...>& P2)
{
  return T(std::get<I>(P1), std::get<I>(P2));
}

template<
    typename T,
    std::size_t... Indices,
    typename... Tp1,
    typename... Tp2>
auto genTuple(std::tuple<Tp1...>& P1,
              std::tuple<Tp2...>& P2,
              std::index_sequence<Indices...>)
{
  return std::make_tuple(genTupleImpl<T, Indices>(P1, P2)...);
}

template<
    typename RT,
    typename T1,
    size_t I,
    typename... Tp2,
    typename... Tp3>
RT genConn1Impl(T1& inputLayer,
                std::tuple<Tp2...>& P2,
                std::tuple<Tp3...>& P3,
                int i,
                int j)
{
  return RT(inputLayer, std::get<I>(P2), std::get<I>(P3), i, j);
}

template<
    typename RT,
    typename T1,
    std::size_t... Indices,
    typename... Tp2,
    typename... Tp3>
auto genConn1(T1& inputLayer,
              std::tuple<Tp2...>& P2,
              std::tuple<Tp3...>& P3,
              int i,
              int j,
              std::index_sequence<Indices...>)
{
  return std::make_tuple(genConn1Impl
                         <RT, T1, Indices>(inputLayer, P2, P3, i, j)...);
}

template<
    typename RT,
    size_t I,
    typename... Tp1,
    typename... Tp2,
    typename... Tp3>
RT genConn2Impl(std::tuple<Tp1...>& P1,
                std::tuple<Tp2...>& P2,
                std::tuple<Tp3...>& P3)
{
  return RT(std::get<I>(P1), std::get<I>(P2), std::get<I>(P3));
}

template<
    typename RT,
    std::size_t... Indices,
    typename... Tp1,
    typename... Tp2,
    typename... Tp3>
auto genConn2(std::tuple<Tp1...>& P1,
              std::tuple<Tp2...>& P2,
              std::tuple<Tp3...>& P3,
              std::index_sequence<Indices...>)
{
  return std::make_tuple(genConn2Impl<RT, Indices>(P1, P2, P3)...);
}

template<
    typename RT, size_t I,
    typename... Tp1,
    typename... Tp2,
    typename... Tp3>
RT genConn5Impl(std::tuple<Tp1...>& P1,
                std::tuple<Tp2...>& P2,
                std::tuple<Tp3...>& P3,
                int i,
                int j)
{
  return RT(std::get<I % 12>(P1), std::get<I / 12>(P2),
            std::get<I>(P3), i, j);
}

template<
    typename RT,
    std::size_t... Indices,
    typename... Tp1,
    typename... Tp2,
    typename... Tp3>
auto genConn5(std::tuple<Tp1...>& P1,
              std::tuple<Tp2...>& P2,
              std::tuple<Tp3...>& P3,
              int i,
              int j,
              std::index_sequence<Indices...>)
{
  return std::make_tuple(genConn5Impl<RT, Indices>(P1, P2, P3, i, j)...);
}

template<
    typename RT,
    typename T1,
    size_t I,
    typename... Tp2,
    typename... Tp3>
RT genLayer1BiasImpl(T1& biasLayer,
                     std::tuple<Tp2...>& P2,
                     std::tuple<Tp3...>& P3)
{
  return RT(biasLayer, std::get<I>(P2), std::get<I>(P3));
}

template<
    typename RT,
    typename T1,
    std::size_t... Indices,
    typename... Tp2,
    typename... Tp3>
auto genLayer1Bias(T1& biasLayer,
                   std::tuple<Tp2...>& P2,
                   std::tuple<Tp3...>& P3,
                   std::index_sequence<Indices...>)
{
  return std::make_tuple(genLayer1BiasImpl
                         <RT, T1, Indices>(biasLayer, P2, P3)...);
}

/**
 * Test the convolutional network on part of MNIST dataset.
 * 1000 cases for training and 100 cases for testing.
 * Raw images in MNIST are grey images 
 * which means all value is in [0, 255].
 * Then they are normalized to [-1, 1]
 * by (grey - 127.5) / 127.5 and stored in 
 * mnist_1000_train_cases_100_test_cases.zip
 *
 * There are many kinds of CNNs on MNIST dataset.
 * Following is LeNet1 which is referred in
 * http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf
 *
 */
BOOST_AUTO_TEST_CASE(LeNet1Test)
{
  
  // Input layer
  NeuronLayer<IdentityFunction, arma::mat> inputLayer(28, 28);
  auto layer0 = std::tie(inputLayer);
  
  /**
   * Bias layer. 
   * Actually this layer is useless, 
   * the bias value are stored in connections connected to this layer.
   * So we just use one bias layer.
   */
  BiasLayer<> biasLayer(1);
  
  // 4 layers in H1
  auto layer1 = genTuple<4>(
      [&](size_t)
      {
        return NeuronLayer<IdentityFunction, arma::mat>(24, 24);
      });
  
  // 4 layers in H2
  auto layer2 = genTuple<4>(
      [&](size_t)
      {
        return NeuronLayer<LogisticFunction, arma::mat>(12, 12);
      });
  
  // 12 layers in H3
  auto layer3 = genTuple<12>(
      [&](size_t)
      {
        return NeuronLayer<IdentityFunction, arma::mat>(8, 8);
      });
  
  // 12 layers in H4
  auto layer4 = genTuple<12>(
      [&](size_t)
      {
        return NeuronLayer<LogisticFunction, arma::mat>(4, 4);
      });

  /**
   * Refer the output layer in the paper above as H5.
   * The connections between H4 and H5 are full connections.
   * Data in H4 is 2-dim, but in H5 1-dim.
   * To make things easier, shared memory is used.
   * layer 5 has 10 neurons, each neuron is regarded as a sub-layer.
   * The 10 sub-layers should share memory with layer 5.
   */
  
  // Shared input storage.
  arma::colvec layer5Input(10);
  auto layer5InputSub = genTuple<10>(
      [&](size_t i)
      {
        return arma::mat(layer5Input.memptr() + i, 1, 1, false, true);
      });
  
  // Shared delta storage.
  arma::colvec layer5Delta(10);
  auto layer5DeltaSub = genTuple<10>(
      [&](size_t i)
      {
        return arma::mat(layer5Delta.memptr() + i, 1, 1, false, true);
      });
  
  // Layer 5 is a softmax layer, which is different from the paper.
  SoftmaxLayer<arma::mat, arma::colvec> layer5_0(layer5Input, layer5Delta);
  auto layer5 = std::tie(layer5_0);
  
  // Sub-layers for layer 5
  auto layer5Sub = genTuple<NeuronLayer<IdentityFunction, arma::mat> >(
      layer5InputSub, layer5DeltaSub, std::make_index_sequence<10>());
  
  /**
   * Define learning rate and momentum for updating.
   * We just use the same one for all connections.
   */
  double lr = 0.005;
  double mom = 0.5;
  
  double lr1 = lr;
  double mom1 = mom;
  
  double lr2 = lr;
  double mom2 = mom;
  
  double lr3 = lr;
  double mom3 = mom;
  
  double lr4 = lr;
  double mom4 = mom;
  
  double lr5 = lr;
  double mom5 = mom;
  
  // 4 convolutional connections between input layer and layer1(H1)
  auto conn1Opt = genTuple<4>(
    [&](size_t)
    {
      return SteepestDescent<arma::mat, arma::mat>(5, 5, lr1, mom1);
    });
  
  auto conn1 = genConn1<
      ConvConnection<
      decltype(inputLayer),
      decltype(std::get<0>(layer1)),
      decltype(std::get<0>(conn1Opt)),
      NguyenWidrowInitialization<arma::mat> >,
      decltype(inputLayer)>(
      inputLayer, layer1, conn1Opt, 5, 5, std::make_index_sequence<4>());
  
  // 4 pooling connections between layer1(H1) and layer2(H2)
  auto conn2Opt = genTuple<4>(
      [&](size_t)
      {
        return SteepestDescent<arma::mat, arma::mat>(1, 2, lr2, mom2);
      });
  
  auto conn2 = genConn2<
      PoolingConnection<
      decltype(std::get<0>(layer1)),
      decltype(std::get<0>(layer2)),
      decltype(std::get<0>(conn2Opt)),
      MeanPooling,
      arma::mat> >(
      layer1, layer2, conn2Opt, std::make_index_sequence<4>());
  
  auto conn3Opt = genTuple<20>(
      [&](size_t)
      {
        return SteepestDescent<arma::mat, arma::mat>(5, 5, lr3, mom3);
      });
  
  /**
   * 20 convolutional connctions between layer2(H2) and layer3(H3).
   * These connections connect layers irregularly, 
   * so we have to write them one by one.
   * In fact I use a small JAVA program to generate skeleton for it.
   */
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_0(std::get<0>(layer2), std::get<0>(layer3),
                    std::get<0>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_1(std::get<0>(layer2), std::get<1>(layer3),
                    std::get<1>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_2(std::get<0>(layer2), std::get<2>(layer3),
                    std::get<2>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_3(std::get<0>(layer2), std::get<4>(layer3),
                    std::get<3>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_4(std::get<0>(layer2), std::get<5>(layer3),
                    std::get<4>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_5(std::get<1>(layer2), std::get<1>(layer3),
                    std::get<5>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_6(std::get<1>(layer2), std::get<2>(layer3),
                    std::get<6>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_7(std::get<1>(layer2), std::get<3>(layer3),
                    std::get<7>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_8(std::get<1>(layer2), std::get<4>(layer3),
                    std::get<8>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_9(std::get<1>(layer2), std::get<5>(layer3),
                    std::get<9>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_10(std::get<2>(layer2), std::get<6>(layer3),
                     std::get<10>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_11(std::get<2>(layer2), std::get<7>(layer3),
                     std::get<11>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_12(std::get<2>(layer2), std::get<8>(layer3),
                     std::get<12>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_13(std::get<2>(layer2), std::get<10>(layer3),
                     std::get<13>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_14(std::get<2>(layer2), std::get<11>(layer3),
                     std::get<14>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_15(std::get<3>(layer2), std::get<7>(layer3),
                     std::get<15>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_16(std::get<3>(layer2), std::get<8>(layer3),
                     std::get<16>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_17(std::get<3>(layer2), std::get<9>(layer3),
                     std::get<17>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_18(std::get<3>(layer2), std::get<10>(layer3),
                     std::get<18>(conn3Opt), 5, 5);
  
  ConvConnection<
  decltype(std::get<0>(layer2)),
  decltype(std::get<0>(layer3)),
  decltype(std::get<0>(conn3Opt)),
  NguyenWidrowInitialization<arma::mat>,
  arma::mat>conn3_19(std::get<3>(layer2), std::get<11>(layer3),
                     std::get<19>(conn3Opt), 5, 5);
  
  auto conn3 = std::tie(conn3_0, conn3_1, conn3_2, conn3_3, conn3_4,
                        conn3_5, conn3_6, conn3_7, conn3_8, conn3_9,
                        conn3_10, conn3_11, conn3_12, conn3_13,
                        conn3_14, conn3_15, conn3_16, conn3_17,
                        conn3_18, conn3_19);
  
  // 12 pooling connections between layer3(H3) and layer4(H4)
  auto conn4Opt = genTuple<12>(
      [&](size_t)
      {
        return SteepestDescent<arma::mat, arma::mat>(1, 2, lr4, mom4);
      });
  
  auto conn4 = genConn2<
      PoolingConnection<
      decltype(std::get<0>(layer3)),
      decltype(std::get<0>(layer4)),
      decltype(std::get<0>(conn4Opt)),
      MeanPooling,
      arma::mat> >(
      layer3, layer4, conn4Opt, std::make_index_sequence<12>());
  
  // 120 full connections between layer4(H4) and 10 sub-layers for layer5
  auto conn5Opt = genTuple<120>(
      [&](size_t i)
      {
        return SteepestDescent<arma::mat, arma::mat>(4, 4, lr5, mom5);
      });
  
  auto conn5 = genConn5<
      ConvConnection<
      decltype(std::get<0>(layer4)),
      decltype(std::get<0>(layer5Sub)),
      decltype(std::get<0>(conn5Opt)),
      NguyenWidrowInitialization<arma::mat>,
      arma::mat> >(
      layer4, layer5Sub, conn5Opt, 4, 4, std::make_index_sequence<120>());
  
  // Bias for neurons in layer1(H1)
  auto layer1biasOpt = genTuple<4>(
      [&](size_t)
      {
        return SteepestDescent<arma::mat, arma::mat>(1, 1, lr, mom);
      });
  
  auto layer1bias = genLayer1Bias<
      BiasConnection<
      decltype(biasLayer),
      decltype(std::get<0>(layer1)),
      decltype(std::get<0>(layer1biasOpt)),
      NguyenWidrowInitialization<arma::mat>,
      arma::mat>,
      decltype(biasLayer) >(
      biasLayer, layer1, layer1biasOpt, std::make_index_sequence<4>());
  
  // Bias for neurons in layer3(H3)
  auto layer3biasOpt = genTuple<12>(
      [&](size_t)
      {
        return SteepestDescent<arma::mat, arma::mat>(1, 1, lr, mom);
      });
  
  auto layer3bias = genLayer1Bias<
      BiasConnection<
      decltype(biasLayer),
      decltype(std::get<0>(layer3)),
      decltype(std::get<0>(layer3biasOpt)),
      NguyenWidrowInitialization<arma::mat>,
      arma::mat>,
      decltype(biasLayer) >(
      biasLayer, layer3, layer3biasOpt, std::make_index_sequence<12>());
  
  // Bias for neurons in layer5(H5)
  SteepestDescent<arma::mat, arma::mat> layer5biasOpt(1, 10, lr, mom);
  
  CNNFullConnection<
      decltype(biasLayer),
      decltype(layer5_0),
      decltype(layer5biasOpt),
      NguyenWidrowInitialization<arma::mat>,
      arma::mat,
      arma::colvec>
      layer5BiasConn(biasLayer, layer5_0, layer5biasOpt);
  
  auto layer5bias = std::tie(layer5BiasConn);
  
  // network module
  auto connection1 = std::tuple_cat(conn1, layer1bias);
  auto connection3 = std::tuple_cat(conn3, layer3bias);
  auto connection5 = std::tuple_cat(conn5, layer5bias);
  auto modules = std::tie(layer0, connection1, layer1, conn2,
                          layer2, connection3, layer3, conn4,
                          layer4, connection5, layer5);
  
  // One-Hot layer for output, different from the paper.
  OneHotLayer<> outputLayer;
  
  CNN<decltype(modules),
      decltype(outputLayer)>
      net(modules, outputLayer);
  
  arma::cube trainData;
  arma::cube testData;
  arma::mat trainLabels;
  arma::mat testLabels;
  
  trainData.load("mnist_train_data_1000.dat");
  trainLabels.load("mnist_train_labels_1000.dat");
  testData.load("mnist_test_data_100.dat");
  testLabels.load("mnist_test_labels_100.dat");
  
  size_t maxEpochs = 100;
  size_t batchSize = 1;
  double tolerance = 0.05;
  
  // The data is already shuffled.
  bool shuffle = false;
  
  Trainer<decltype(net)> trainer(net, maxEpochs, batchSize, tolerance, shuffle);
  
  // Train network
  trainer.Train(trainData, trainLabels, testData, testLabels);
  
  // Calculate the accuracy in testing dataset.
  double accuracy = 0;
  for (size_t i = 0; i < testData.n_slices; ++i)
  {
    arma::colvec pred;
    net.Predict(testData.slice(i), pred);
    if (sum(pred != testLabels.unsafe_col(i)) == 0)
      accuracy++;
  }
  accuracy /= testData.n_slices;
  
  // Make sure the network converges on training dataset.
  BOOST_REQUIRE_GE(0.05, trainer.TrainingError());
  
  /**
   * Make sure the network achieve high recognization accuracy
   * on testing dataset.
   * In fact, the accuracy is 97% or so.
   */
  BOOST_REQUIRE_GE(accuracy, 0.95);
  
}

BOOST_AUTO_TEST_SUITE_END();
