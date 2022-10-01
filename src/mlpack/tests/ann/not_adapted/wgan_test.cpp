/**
 * @file tests/wgan_test.cpp
 * @author Shikhar Jaiswal
 *
 * Tests the WGAN network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/earth_mover_distance.hpp>
#include <mlpack/methods/ann/gan/gan.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <ensmallen.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace std::placeholders;

/*
 * Tests the standard WGAN implementation on the MNIST dataset.
 * It's not viable to train on bigger parameters due to time constraints.
 * Please refer mlpack/models repository for the tutorial.
 */
TEST_CASE("WGANMNISTTest", "[WGANNetworkTest]")
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
  double clippingParameter = 0.01;

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

  // Create the Discriminator network.
  FFN<EarthMoverDistance<> > discriminator;
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
  discriminator.Add<SigmoidLayer<> >();

  // Create the Generator network.
  FFN<EarthMoverDistance<> > generator;
  generator.Add<TransposedConvolution<> >(noiseDim, 8 * dNumKernels, 2, 2,
      1, 1, 0, 0, 1, 1, 2, 2);
  generator.Add<BatchNorm<> >(1024);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(8 * dNumKernels, 4 * dNumKernels,
      2, 2, 1, 1, 0, 0, 2, 2, 3, 3);
  generator.Add<BatchNorm<> >(1152);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(4 * dNumKernels, 2 * dNumKernels,
      5, 5, 2, 2, 1, 1, 3, 3, 7, 7);
  generator.Add<BatchNorm<> >(3136);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(2 * dNumKernels, dNumKernels, 4, 4,
      2, 2, 1, 1, 7, 7, 14, 14);
  generator.Add<BatchNorm<> >(6272);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(dNumKernels, 1, 4, 4, 2, 2, 1, 1,
      14, 14, 28, 28);
  generator.Add<TanHLayer<> >();

  // Create WGAN.
  GaussianInitialization gaussian(0, 1);
  ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, eps, numIterations,
      tolerance, shuffle);
  std::function<double()> noiseFunction = [] () {
      return RandNormal(0, 1);};
  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGAN> wgan(generator, discriminator, gaussian,
      noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier, clippingParameter);

  Log::Info << "Training..." << std::endl;
  double objVal = wgan.Train(trainData, optimizer);

  // Test that objective value returned by GAN::Train() is finite.
  REQUIRE(std::isfinite(objVal) == true);

  // Generate samples.
  Log::Info << "Sampling..." << std::endl;
  arma::mat noise(noiseDim, batchSize);
  size_t dim = std::sqrt(trainData.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);

  for (size_t i = 0; i < numSamples; ++i)
  {
    arma::mat samples;
    noise.imbue( [&]() { return noiseFunction(); } );

    wgan.Generator().Forward(noise, samples);
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;

    samples = trainData.col(RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim,
        i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
  }

  Log::Info << "Output generated!" << std::endl;

  // Check that Serialization is working correctly.
  arma::mat orgPredictions;
  wgan.Predict(noise, orgPredictions);

  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGAN> wganText(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGAN> wganXml(generator, discriminator, gaussian,
      noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGAN> wganBinary(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  SerializeObjectAll(wgan, wganXml, wganText, wganBinary);

  arma::mat predictions, xmlPredictions, textPredictions, binaryPredictions;
  wgan.Predict(noise, predictions);
  wganXml.Predict(noise, xmlPredictions);
  wganText.Predict(noise, textPredictions);
  wganBinary.Predict(noise, binaryPredictions);

  CheckMatrices(orgPredictions, predictions);
  CheckMatrices(orgPredictions, xmlPredictions);
  CheckMatrices(orgPredictions, textPredictions);
  CheckMatrices(orgPredictions, binaryPredictions);
}

/*
 * Tests the gradient-penalized WGAN implementation on the MNIST dataset.
 * It's not viable to train on bigger parameters due to time constraints.
 * Please refer mlpack/models repository for the tutorial.
 */
TEST_CASE("WGANGPMNISTTest", "[WGANNetworkTest]")
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
  double clippingParameter = 0.01;
  double lambda = 10.0;

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

  // Create the Discriminator network.
  FFN<EarthMoverDistance<> > discriminator;
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
  discriminator.Add<SigmoidLayer<> >();

  // Create the Generator network.
  FFN<EarthMoverDistance<> > generator;
  generator.Add<TransposedConvolution<> >(noiseDim, 8 * dNumKernels, 2, 2,
      1, 1, 0, 0, 1, 1, 2, 2);
  generator.Add<BatchNorm<> >(1024);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(8 * dNumKernels, 4 * dNumKernels,
      2, 2, 1, 1, 0, 0, 2, 2, 3, 3);
  generator.Add<BatchNorm<> >(1152);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(4 * dNumKernels, 2 * dNumKernels,
      5, 5, 2, 2, 1, 1, 3, 3, 7, 7);
  generator.Add<BatchNorm<> >(3136);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(2 * dNumKernels, dNumKernels, 4, 4,
      2, 2, 1, 1, 7, 7, 14, 14);
  generator.Add<BatchNorm<> >(6272);
  generator.Add<ReLULayer<> >();
  generator.Add<TransposedConvolution<> >(dNumKernels, 1, 4, 4, 2, 2, 1, 1,
      14, 14, 28, 28);
  generator.Add<TanHLayer<> >();

  // Create WGANGP.
  GaussianInitialization gaussian(0, 1);
  ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, eps, numIterations,
      tolerance, shuffle);
  std::function<double()> noiseFunction = [] () {
      return RandNormal(0, 1);};
  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGANGP> wganGP(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier, clippingParameter, lambda);

  Log::Info << "Training..." << std::endl;
  double objVal = wganGP.Train(trainData, optimizer);

  // Test that objective value returned by GAN::Train() is finite.
  REQUIRE(std::isfinite(objVal) == true);

  // Generate samples.
  Log::Info << "Sampling..." << std::endl;
  arma::mat noise(noiseDim, batchSize);
  size_t dim = std::sqrt(trainData.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);

  for (size_t i = 0; i < numSamples; ++i)
  {
    arma::mat samples;
    noise.imbue( [&]() { return noiseFunction(); } );

    wganGP.Generator().Forward(noise, samples);
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;

    samples = trainData.col(RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim,
        i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
  }

  Log::Info << "Output generated!" << std::endl;

  // Check that Serialization is working correctly.
  arma::mat orgPredictions;
  wganGP.Predict(noise, orgPredictions);

  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGANGP> wganGPJson(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGANGP> wganGPXml(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  GAN<FFN<EarthMoverDistance<> >, GaussianInitialization,
      std::function<double()>, WGANGP> wganGPBinary(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

  SerializeObjectAll(wganGP, wganGPXml, wganGPJson, wganGPBinary);

  arma::mat predictions, xmlPredictions, jsonPredictions, binaryPredictions;
  wganGP.Predict(noise, predictions);
  wganGPXml.Predict(noise, xmlPredictions);
  wganGPJson.Predict(noise, jsonPredictions);
  wganGPBinary.Predict(noise, binaryPredictions);

  CheckMatrices(orgPredictions, predictions);
  CheckMatrices(orgPredictions, xmlPredictions);
  CheckMatrices(orgPredictions, jsonPredictions);
  CheckMatrices(orgPredictions, binaryPredictions);
}
