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
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::optimization;
using namespace mlpack::regression;
using namespace std::placeholders;

BOOST_AUTO_TEST_SUITE(GANNetworkTest);


BOOST_AUTO_TEST_CASE(GanTest)
{
  size_t hiddenLayerSize1 = 1024;
  size_t gOutputSize;
  size_t dOutputSize = 1;
  size_t batchSize = 100;
  size_t noiseDim  = 100;

  // Load the dataset
  arma::mat trainData, dataset, noiseData;
  trainData.load("train4.txt");
  trainData = trainData.cols(1, 1000);
  noiseData.set_size(100, trainData.n_cols);
  noiseData.imbue([&]() { return arma::as_scalar(RandNormal(0, 1));});
  std::cout << arma::size(trainData) << std::endl;
  gOutputSize = trainData.n_rows;
  // Discriminator network
  FFN<CrossEntropyError<>> discriminator;
  discriminator.Add<Linear<>>(trainData.n_cols, hiddenLayerSize1);
  discriminator.Add<LeakyReLU<>>(0.2);
  discriminator.Add<Linear<>>(hiddenLayerSize1, hiddenLayerSize1 / 2);
  discriminator.Add<LeakyReLU<>>(0.2);
  discriminator.Add<Linear<>>(hiddenLayerSize1 / 2, hiddenLayerSize1 / 4);
  discriminator.Add<LeakyReLU<>>(0.2);
  discriminator.Add<Linear<>>(hiddenLayerSize1 / 4, 1);
  discriminator.Add<SigmoidLayer<>>();

  // Generator network
  FFN<CrossEntropyError<>> generator;
  generator.Add<Linear<>>(noiseDim, hiddenLayerSize1 / 4);
  generator.Add<LeakyReLU<>>(0.2);
  generator.Add<Linear<>>(hiddenLayerSize1 / 4, hiddenLayerSize1 / 2);
  generator.Add<LeakyReLU<>>(0.2);
  generator.Add<Linear<>>(hiddenLayerSize1 / 2, trainData.n_cols);
  generator.Add<SigmoidLayer<>>();

  // Intialisation function
  GaussianInitialization gaussian(0, 1);
  // Optimizer
  MiniBatchSGD optimizer(batchSize, 1e-4, 100 * trainData.n_cols, 1e-5, true);

  std::normal_distribution<> noiseFunction(0.0, 1.0);
  // GAN model
  GAN<> gan(trainData, generator, discriminator, gaussian, noiseFunction,
      trainData.n_rows, batchSize, 10, 10);
  gan.Train(optimizer);

  // Generate samples
  Log::Info << "Sampling..." << std::endl;
  arma::mat noise(noiseDim, 1);
  size_t dim = std::sqrt(trainData.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);
  for (size_t i = 0; i < numSamples; i++)
  {
    arma::mat samples;
    noise.imbue( [&]() { return math::Random(0, 1); } );
    generator.Forward(noise, samples);

    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;


    samples = trainData.col(math::RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim, i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
  }
  std::string output_dataset = "./output_gan_ffn"
  Log::Info << "Saving output to " << output_dataset << "..." << std::endl;
  generatedData.save(output_dataset, arma::raw_ascii);
  Log::Info << "Output saved!" << std::endl;
}
BOOST_AUTO_TEST_SUITE_END();
