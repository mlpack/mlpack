/**
 * @file gan_network_test.cpp
 * @author Kris Singh
 *
 * Tests the gan Network
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/gan.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::optimization;
using namespace mlpack::regression;
using namespace std::placeholders;

BOOST_AUTO_TEST_SUITE(GANNetworkTest);

BOOST_AUTO_TEST_CASE(GaussianDistributionTest)
{
  // Create 1000 samples of N(-1, 1)
  // Create 1000 samples of N(-1, 1)
  size_t discOutputSize = 1;
  size_t gOutputSize = 1;
  size_t hiddenLayerSize1 = 6;
  size_t hiddenLayerSize2 = 5;
  size_t batchSize = 200;
  arma::mat trainData(1, 1000), noiseData(1, 1000);
  double mean = -1;
  double variance = -1;
  noiseData.imbue([&]() { return arma::as_scalar(Random(-1, 1));});
  trainData.imbue( [&]() { return arma::as_scalar(RandNormal(mean, variance));});
  // trainData.save('trainData.txt', arma::raw_ascii);

  // Create the Discrminator network
  FFN<CrossEntropyError<>> discriminator;
  discriminator.Add<Linear<>> (gOutputSize, hiddenLayerSize1);
  discriminator.Add<TanHLayer<>>();
  discriminator.Add<Linear<>> (hiddenLayerSize1, hiddenLayerSize2);
  discriminator.Add<TanHLayer<>>();
  discriminator.Add<Linear<>> (hiddenLayerSize2, discOutputSize);
  discriminator.Add<SigmoidLayer<>>();
  // Create the Generator network
  FFN<CrossEntropyError<>> generator;
  generator = discriminator;

  // Shuffle the input
  trainData = arma::shuffle(trainData);
  MomentumUpdate momentum(0.6);
  AdamType<> optimizer(0.001, 0.9, 0.999, 1e-5, trainData.n_cols * 10, true);
  // Create Gan
  GaussianInitialization gaussian(0, 0.1);
  GAN<> gan(trainData, noiseData, generator, discriminator, gaussian,
      batchSize, 10);
  gan.Train(optimizer);
  arma::mat result;
  arma::mat noise(1, 1);
  arma::mat generatedData(gOutputSize,10);
  for (size_t i = 0; i < 10; i++)
  {
    noise(0) = Random(-1, 1);
    generator.Forward(noise, result);
    generatedData.col(i) = result;
  }
}


BOOST_AUTO_TEST_CASE(GanTest)
{
  size_t hiddenLayerSize1 = 128;
  size_t gOutputSize;
  size_t dOutputSize = 1;
  size_t batchSize = 100;
  // Load the dataset
  arma::mat trainData, dataset, noiseData;
  trainData.load("train4.txt");
  std::cout << arma::size(trainData) << std::endl;
  noiseData.set_size(100, trainData.n_cols);
  noiseData.imbue([&]() { return arma::as_scalar(RandNormal(0, 1));});
  std::cout << arma::size(trainData) << std::endl;
  gOutputSize = trainData.n_rows;
  // Discriminator network
  FFN<CrossEntropyError<> > discriminator;
  discriminator.Add<Linear<> >(gOutputSize, hiddenLayerSize1);
  discriminator.Add<ReLULayer<>>();
  discriminator.Add<Linear<> >(hiddenLayerSize1, dOutputSize);
  discriminator.Add<SigmoidLayer<> >();

  // Generator network
  FFN<CrossEntropyError<>> generator;
  generator.Add<Linear<> >(noiseData.n_rows, hiddenLayerSize1);
  generator.Add<ReLULayer<> >();
  generator.Add<Linear<> >(hiddenLayerSize1, gOutputSize);
  generator.Add<SigmoidLayer<> >();

  // Intialisation function
  GaussianInitialization gaussian(0, 0.1);
  // Optimizer
  AdamType<> optimizer(0.001, 0.9, 0.999, 1e-5, trainData.n_cols * 20, true);
  // GAN model
  GAN<> gan(trainData, noiseData, generator, discriminator, gaussian,
      batchSize, 10);

  gan.Train(optimizer);

  // Generate samples
  arma::mat samples;
  arma::mat noise(100, 1);
  arma::mat generatedData(gOutputSize,10);
  for (size_t i = 0; i < 10; i++)
  {
    noise.imbue([&]() { return arma::as_scalar(RandNormal(0, 1));});
    generator.Forward(noise, samples);
    generatedData.col(i) = samples;
  }
}
BOOST_AUTO_TEST_SUITE_END();
