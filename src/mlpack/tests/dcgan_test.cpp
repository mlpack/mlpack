/**
 * @file dcgan_test.cpp
 * @author Shikhar Jaiswal
 *
 * Tests the DCGAN network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
#include <mlpack/methods/ann/gan/gan.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::regression;
using namespace std::placeholders;

BOOST_AUTO_TEST_SUITE(DCGANNetworkTest);

/*
 * Tests the DCGAN implementation on the MNIST dataset.
 * It's not viable to train on bigger parameters due to time constraints.
 * Please refer mlpack/models repository for the tutorial.
 */
BOOST_AUTO_TEST_CASE(DCGANMNISTTest)
{
  size_t dNumKernels = 32;
  size_t discriminatorPreTrain = 5;
  size_t batchSize = 5;
  size_t noiseDim = 100;
  size_t generatorUpdateStep = 1;
  size_t numSamples = 10;
  double stepSize = 0.0003;
  double eps = 1e-8;
  size_t numEpoches = 1;
  double tolerance = 1e-5;
  int datasetMaxCols = 10;
  bool shuffle = true;
  double multiplier = 10;

  Log::Info << std::boolalpha
      << " batchSize = " << batchSize << std::endl
      << " generatorUpdateStep = " << generatorUpdateStep << std::endl
      << " noiseDim = " << noiseDim << std::endl
      << " numSamples = " << numSamples << std::endl
      << " stepSize = " << stepSize << std::endl
      << " numEpoches = " << numEpoches << std::endl
      << " tolerance = " << tolerance << std::endl
      << " shuffle = " << shuffle << std::endl;

  arma::mat trainData;
  trainData.load("mnist_first250_training_4s_and_9s.arm");
  Log::Info << arma::size(trainData) << std::endl;

  trainData = trainData.cols(0, datasetMaxCols - 1);

  size_t numIterations = trainData.n_cols * numEpoches;
  numIterations /= batchSize;

  Log::Info << "Dataset loaded (" << trainData.n_rows << ", "
            << trainData.n_cols << ")" << std::endl;
  Log::Info << trainData.n_rows << "--------" << trainData.n_cols << std::endl;

  // Create the Discriminator network
  FFN<SigmoidCrossEntropyError<> > discriminator;
  discriminator.Add<Convolution<> >(1, dNumKernels, 4, 4, 2, 2, 1, 1, 28, 28);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(dNumKernels, 2 * dNumKernels, 4, 4, 2, 2,
      1, 1, 14, 14);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(2 * dNumKernels, 4 * dNumKernels, 4, 4,
      2, 2, 1, 1, 7, 7);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(4 * dNumKernels, 8 * dNumKernels, 4, 4,
      2, 2, 2, 2, 3, 3);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(8 * dNumKernels, 1, 4, 4, 1, 1,
      1, 1, 2, 2);

  // Create the Generator network
  FFN<SigmoidCrossEntropyError<> > generator;
  generator.Add<TransposedConvolution<> >(noiseDim, 8 * dNumKernels, 2, 2,
      1, 1, 1, 1, 1, 1);
  generator.Add<BatchNorm<> >(1024);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(8 * dNumKernels, 4 * dNumKernels,
      2, 2, 1, 1, 0, 0, 2, 2);
  generator.Add<BatchNorm<> >(1152);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(4 * dNumKernels, 2 * dNumKernels,
      5, 5, 2, 2, 1, 1, 3, 3);
  generator.Add<BatchNorm<> >(3136);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(2 * dNumKernels, dNumKernels, 8, 8,
      1, 1, 1, 1, 7, 7);
  generator.Add<BatchNorm<> >(6272);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(dNumKernels, 1, 15, 15, 1, 1, 1, 1,
      14, 14);
  generator.Add<TanHLayer<> >();

  // Create DCGAN
  GaussianInitialization gaussian(0, 1);
  ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, eps, numIterations,
      tolerance, shuffle);
  std::function<double()> noiseFunction = [] () {
      return math::RandNormal(0, 1);};
  GAN<FFN<CrossEntropyError<> >, GaussianInitialization,
      std::function<double()>, DCGAN> dcgan(generator, discriminator, gaussian,
      noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  Log::Info << "Training..." << std::endl;
  double objVal = dcgan.Train(trainData, optimizer);

  // Test that objective value returned by GAN::Train() is finite.
  BOOST_REQUIRE_EQUAL(std::isfinite(objVal), true);

  // Generate samples.
  Log::Info << "Sampling..." << std::endl;
  arma::mat noise(noiseDim, 1);
  size_t dim = std::sqrt(trainData.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);

  for (size_t i = 0; i < numSamples; i++)
  {
    arma::mat samples;
    noise.imbue( [&]() { return noiseFunction(); } );

    dcgan.Generator().Forward(noise, samples);
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;

    samples = trainData.col(math::RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim,
        i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
  }

  Log::Info << "Output generated!" << std::endl;

  // Check that Serialization is working correctly.
  arma::mat orgPredictions;
  dcgan.Predict(noise, orgPredictions);

  GAN<FFN<CrossEntropyError<> >, GaussianInitialization,
      std::function<double()>, DCGAN> dcganText(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  GAN<FFN<CrossEntropyError<> >, GaussianInitialization,
      std::function<double()>, DCGAN> dcganXml(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  GAN<FFN<CrossEntropyError<> >, GaussianInitialization,
      std::function<double()>, DCGAN> dcganBinary(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  SerializeObjectAll(dcgan, dcganXml, dcganText, dcganBinary);

  arma::mat predictions, xmlPredictions, textPredictions, binaryPredictions;
  dcgan.Predict(noise, predictions);
  dcganXml.Predict(noise, xmlPredictions);
  dcganText.Predict(noise, textPredictions);
  dcganBinary.Predict(noise, binaryPredictions);

  CheckMatrices(orgPredictions, predictions);
  CheckMatrices(orgPredictions, xmlPredictions);
  CheckMatrices(orgPredictions, textPredictions);
  CheckMatrices(orgPredictions, binaryPredictions);
}

/*
 * Tests the DCGAN implementation on the CelebA dataset.
 * It's currently not possible to run this every time due to time constraints.
 * Please refer mlpack/models repository for the tutorial.

BOOST_AUTO_TEST_CASE(DCGANCelebATest)
{
  size_t dNumKernels = 64;
  size_t discriminatorPreTrain = 300;
  size_t batchSize = 1;
  size_t noiseDim = 100;
  size_t generatorUpdateStep = 1;
  size_t numSamples = 10;
  double stepSize = 0.0003;
  double eps = 1e-8;
  size_t numEpoches = 20;
  double tolerance = 1e-5;
  int datasetMaxCols = -1;
  bool shuffle = true;
  double multiplier = 10;

  Log::Info << std::boolalpha
      << " batchSize = " << batchSize << std::endl
      << " generatorUpdateStep = " << generatorUpdateStep << std::endl
      << " noiseDim = " << noiseDim << std::endl
      << " numSamples = " << numSamples << std::endl
      << " stepSize = " << stepSize << std::endl
      << " numEpoches = " << numEpoches << std::endl
      << " tolerance = " << tolerance << std::endl
      << " shuffle = " << shuffle << std::endl;

  arma::mat trainData;
  trainData.load("celeba.csv");
  Log::Info << arma::size(trainData) << std::endl;

  if (datasetMaxCols > 0)
    trainData = trainData.cols(0, datasetMaxCols - 1);

  size_t numIterations = trainData.n_cols * numEpoches;
  numIterations /= batchSize;

  Log::Info << "Dataset loaded (" << trainData.n_rows << ", "
            << trainData.n_cols << ")" << std::endl;
  Log::Info << trainData.n_rows << "--------" << trainData.n_cols << std::endl;

  // Create the Discriminator network
  FFN<SigmoidCrossEntropyError<> > discriminator;
  discriminator.Add<Convolution<> >(3, dNumKernels, 4, 4, 2, 2, 1, 1, 64, 64);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(dNumKernels, 2 * dNumKernels, 4, 4, 2, 2,
      1, 1, 32, 32);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(2 * dNumKernels, 4 * dNumKernels, 4, 4,
      2, 2, 1, 1, 16, 16);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(4 * dNumKernels, 8 * dNumKernels, 4, 4,
      2, 2, 1, 1, 8, 8);
  discriminator.Add<LeakyReLU<> >(0.2);
  discriminator.Add<Convolution<> >(8 * dNumKernels, 1, 4, 4, 1, 1,
      0, 0, 4, 4);

  // Create the Generator network
  FFN<SigmoidCrossEntropyError<> > generator;
  generator.Add<TransposedConvolution<> >(noiseDim, 8 * dNumKernels, 4, 4,
      1, 1, 2, 2, 1, 1);
  generator.Add<BatchNorm<> >(4096);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(8 * dNumKernels, 4 * dNumKernels,
      5, 5, 1, 1, 1, 1, 4, 4);
  generator.Add<BatchNorm<> >(8192);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(4 * dNumKernels, 2 * dNumKernels,
      9, 9, 1, 1, 1, 1, 8, 8);
  generator.Add<BatchNorm<> >(16384);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(2 * dNumKernels, dNumKernels, 17, 17,
      1, 1, 1, 1, 16, 16);
  generator.Add<BatchNorm<> >(32768);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(dNumKernels, 3, 33, 33, 1, 1, 1, 1,
      32, 32);
  generator.Add<TanHLayer<> >();

  // Create DCGAN
  GaussianInitialization gaussian(0, 1);
  ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, eps, numIterations,
      tolerance, shuffle);
  std::function<double()> noiseFunction = [] () {
      return math::RandNormal(0, 1);};
  GAN<FFN<SigmoidCrossEntropyError<> >, GaussianInitialization,
      std::function<double()>, DCGAN> dcgan(trainData, generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  Log::Info << "Training..." << std::endl;
  dcgan.Train(optimizer);

  // Generate samples
  Log::Info << "Sampling..." << std::endl;
  arma::mat noise(noiseDim, 1);
  size_t dim = std::sqrt(trainData.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);

  for (size_t i = 0; i < numSamples; i++)
  {
    arma::mat samples;
    noise.imbue( [&]() { return noiseFunction(); } );

    dcgan.Generator().Forward(noise, samples);
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;

    samples = trainData.col(math::RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim,
        i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
  }

  Log::Info << "Output generated!" << std::endl;
}
*/

BOOST_AUTO_TEST_SUITE_END();
