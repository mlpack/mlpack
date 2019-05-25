/**
 * @file cyclegan_test.cpp
 * @author Shikhar Jaiswal
 *
 * Tests the CycleGAN network.
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

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::regression;
using namespace std::placeholders;

BOOST_AUTO_TEST_SUITE(CycleGANNetworkTest);

/*
 * Tests the CycleGAN implementation on MNIST to SVHN conversion.
 * It's not viable to train on bigger parameters due to time constraints.
 * Please refer mlpack/models repository for the tutorial.
 *
BOOST_AUTO_TEST_CASE(CycleGANMNISTToSVHNTest)
{
  size_t dNumKernels = 64;
  size_t discriminatorPreTrain = 5;
  size_t batchSize = 5;
  size_t generatorUpdateStep = 1;
  size_t numSamples = 10;
  double stepSize = 0.0003;
  double eps = 1e-8;
  size_t numEpoches = 1;
  double tolerance = 1e-5;
  int datasetMaxCols = 10;
  bool shuffle = true;
  double lambda = 10.0;
  double multiplier = 10.0;

  Log::Info << std::boolalpha
      << " batchSize = " << batchSize << std::endl
      << " generatorUpdateStep = " << generatorUpdateStep << std::endl
      << " numSamples = " << numSamples << std::endl
      << " stepSize = " << stepSize << std::endl
      << " numEpoches = " << numEpoches << std::endl
      << " tolerance = " << tolerance << std::endl
      << " shuffle = " << shuffle << std::endl;

  arma::mat trainData;
  trainData.load("");
  Log::Info << arma::size(trainData) << std::endl;

  trainData = trainData.cols(0, datasetMaxCols - 1);

  size_t numIterations = trainData.n_cols * numEpoches;
  numIterations /= batchSize;

  Log::Info << "Dataset loaded (" << trainData.n_rows << ", "
            << trainData.n_cols << ")" << std::endl;
  Log::Info << trainData.n_rows << "--------" << trainData.n_cols << std::endl;

  // Create the discriminator network for MNIST
  FFN<SigmoidCrossEntropyError<> > discriminatorX;
  discriminatorX.Add<Convolution<> >(1, dNumKernels, 4, 4, 2, 2, 3, 3, 28, 28);
  discriminatorX.Add<LeakyReLU<> >(0.05);
  discriminatorX.Add<Convolution<> >(dNumKernels, 2 * dNumKernels, 4, 4, 2, 2,
      1, 1, 16, 16);
  discriminatorX.Add<LeakyReLU<> >(0.05);
  discriminatorX.Add<Convolution<> >(2 * dNumKernels, 4 * dNumKernels, 4, 4,
      2, 2, 1, 1, 8, 8);
  discriminatorX.Add<LeakyReLU<> >(0.05);
  discriminatorX.Add<Convolution<> >(4 * dNumKernels, 1, 4, 4, 1, 1, 0, 0, 4, 4);

  // Create the generator network for transfering from MNIST to SVHN.
  FFN<SigmoidCrossEntropyError<> > generatorX;
  // Encoding blocks.
  generatorX.Add<Convolution<> >(1, dNumKernels, 4, 4, 2, 2, 3, 3, 28, 28);
  generatorX.Add<LeakyReLU<> >(0.05);
  generatorX.Add<Convolution<> >(dNumKernels, 2 * dNumKernels, 4, 4, 2, 2, 1, 1,
      16, 16);
  generatorX.Add<LeakyReLU<> >(0.05);
  // Residual blocks.
  generatorX.Add<Convolution<> >(2 * dNumKernels, 2 * dNumKernels, 3, 3, 1, 1,
      1, 1, 8, 8);
  generatorX.Add<LeakyReLU<> >(0.05);
  generatorX.Add<Convolution<> >(2 * dNumKernels, 2 * dNumKernels, 3, 3, 1, 1,
      1, 1, 8, 8);
  generatorX.Add<LeakyReLU<> >(0.05);
  // Decoding blocks.
  generatorX.Add<TransposedConvolution<> >(2 * dNumKernels, dNumKernels, 9, 9,
      1, 1, 8, 8, 8, 8);
  generatorX.Add<LeakyReLU<> >(0.05);
  generatorX.Add<TransposedConvolution<> >(dNumKernels, 3, 17, 17, 1, 1, 16, 16,
      16, 16);
  generatorX.Add<TanHLayer<> >();

  // Create the discriminator network for SVHN.
  FFN<SigmoidCrossEntropyError<> > discriminatorY;
  discriminatorY.Add<Convolution<> >(3, dNumKernels, 4, 4, 2, 2, 1, 1, 32, 32);
  discriminatorY.Add<LeakyReLU<> >(0.05);
  discriminatorY.Add<Convolution<> >(dNumKernels, 2 * dNumKernels, 4, 4, 2, 2,
      1, 1, 16, 16);
  discriminatorY.Add<LeakyReLU<> >(0.05);
  discriminatorY.Add<Convolution<> >(2 * dNumKernels, 4 * dNumKernels, 4, 4,
      2, 2, 1, 1, 8, 8);
  discriminatorY.Add<LeakyReLU<> >(0.05);
  discriminatorY.Add<Convolution<> >(4 * dNumKernels, 1, 4, 4, 1, 1, 0, 0, 4, 4);

  // Create the generator network for transfering from SVHN to MNIST.
  FFN<SigmoidCrossEntropyError<> > generatorY;
  // Encoding blocks.
  generatorY.Add<Convolution<> >(3, dNumKernels, 4, 4, 2, 2, 1, 1, 32, 32);
  generatorY.Add<LeakyReLU<> >(0.05);
  generatorY.Add<Convolution<> >(dNumKernels, 2 * dNumKernels, 4, 4, 2, 2, 1, 1,
      16, 16);
  generatorY.Add<LeakyReLU<> >(0.05);
  // Residual blocks.
  generatorY.Add<Convolution<> >(2 * dNumKernels, 2 * dNumKernels, 3, 3, 1, 1,
      1, 1, 8, 8);
  generatorY.Add<LeakyReLU<> >(0.05);
  generatorY.Add<Convolution<> >(2 * dNumKernels, 2 * dNumKernels, 3, 3, 1, 1,
      1, 1, 8, 8);
  generatorY.Add<LeakyReLU<> >(0.05);
  // Decoding blocks.
  generatorY.Add<TransposedConvolution<> >(2 * dNumKernels, dNumKernels, 9, 9,
      1, 1, 8, 8, 8, 8);
  generatorY.Add<LeakyReLU<> >(0.05);
  generatorY.Add<TransposedConvolution<> >(dNumKernels, 1, 13, 13, 1, 1, 12, 12,
      16, 16);
  generatorY.Add<TanHLayer<> >();

  // Create CycleGAN
  GaussianInitialization gaussian(0, 1);
  Adam optimizer(stepSize, batchSize, 0.9, 0.999, eps, numIterations,
      tolerance, shuffle);
  CycleGAN<FFN<SigmoidCrossEntropyError<> >, GaussianInitialization> cyclegan(trainDataX,
      trainDataY, generatorX, generatorY, discriminatorX, discriminatorY,        
      gaussian, batchSize, generatorUpdateStep, discriminatorPreTrain, lambda,
      multiplier);

  Log::Info << "Training..." << std::endl;
  double objVal = cyclegan.Train(optimizer);
  BOOST_REQUIRE_EQUAL(std::isfinite(objVal), true);

  // Generate samples
  Log::Info << "Sampling..." << std::endl;
  size_t dim = std::sqrt(trainData.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);

  for (size_t i = 0; i < numSamples; i++)
  {
    arma::mat samples;

    cyclegan.Generator().Forward(input, samples);
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
