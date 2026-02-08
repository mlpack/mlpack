/**
 * @file tests/ann/layer/conv_deconv_autoencoder_test.cpp
 * @author Ranjodh Singh
 *
 * Test the Convolution and Transposed Convolution layers by making an autoencoder.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/data/text_options.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/transposed_convolution.hpp>

#include "../catch.hpp"

using namespace mlpack;

// Calculates mean loss over batches.
template<typename NetworkType,
         typename DataType>
double MeanTestLoss(NetworkType& model, DataType& testSet, size_t batchSize)
{
  double loss = 0;
  size_t nofPoints = testSet.n_cols;

  size_t i;
  for (i = 0; i < (size_t) nofPoints / batchSize; i++)
  {
    loss += model.Evaluate(
        testSet.cols(batchSize * i, batchSize * (i + 1) - 1),
        testSet.cols(batchSize * i, batchSize * (i + 1) - 1));
  }

  if (nofPoints % batchSize != 0)
  {
    loss += model.Evaluate(testSet.cols(batchSize * i, nofPoints - 1),
                           testSet.cols(batchSize * i, nofPoints - 1));
    loss /= nofPoints / batchSize + 1;
  }
  else
  {
    loss /= nofPoints / batchSize;
  }

  return loss;
}

/**
 * Create a simple autoencoder with Convolution and Transposed Convolution
 * layers, train it on a subset of MNIST, and check the mean squared error.
 */
TEST_CASE("ConvTransConvAutoencoderTest", "[AutoEncoderTest]")
{
  FFN<MeanSquaredError, XavierInitialization> autoencoder;
  autoencoder.Add<Convolution>(16, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<LeakyReLU>();
  autoencoder.Add<Convolution>(32, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<LeakyReLU>();
  autoencoder.Add<TransposedConvolution>(16, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<LeakyReLU>();
  autoencoder.Add<TransposedConvolution>(1, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<Sigmoid>();
  autoencoder.InputDimensions() = std::vector<size_t>({28, 28});

  TextOptions opts;
  opts.Fatal() = true;
  opts.NoTranspose() = true;

  arma::mat data;
  Load("mnist_first250_training_4s_and_9s.csv", data, opts);
  data = (data - data.min()) / (data.max() - data.min());

  arma::mat trainData, testData;
  Split(data, trainData, testData, 0.2);

  ens::Adam optimizer;
  optimizer.StepSize() = 0.1;
  optimizer.BatchSize() = 16;
  optimizer.MaxIterations() = 2 * trainData.n_cols;
  autoencoder.Train(trainData, trainData, optimizer);

  REQUIRE(MeanTestLoss<>(autoencoder, testData, optimizer.BatchSize()) < 0.1);
}
