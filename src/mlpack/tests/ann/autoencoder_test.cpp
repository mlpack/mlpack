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
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/transposed_convolution.hpp>

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

// Calculates mean loss over batches.
template<typename NetworkType = FFN<MeanSquaredError, HeInitialization>,
         typename DataType = arma::mat>
double MeanTestLoss(NetworkType& model, DataType& testSet, size_t batchSize)
{
  double loss = 0;
  size_t nofPoints = testSet.n_cols;
  size_t i;

  for (i = 0; i < (size_t) nofPoints / batchSize; i++)
  {
    loss +=
        model.Evaluate(testSet.cols(batchSize * i, batchSize * (i + 1) - 1),
                       testSet.cols(batchSize * i, batchSize * (i + 1) - 1));
  }

  if (nofPoints % batchSize != 0)
  {
    loss += model.Evaluate(testSet.cols(batchSize * i, nofPoints - 1),
                           testSet.cols(batchSize * i, nofPoints - 1));
    loss /= (int) nofPoints / batchSize + 1;
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

  arma::mat data;
  data::Load("mnist_first250_training_4s_and_9s.csv", data, true, false);
  // min max normalization
  data = (data - min(min(data))) / (max(max(data) - min(min(data))));

  arma::mat trainData, testData;
  data::Split(data, trainData, testData, 0.1);

  ens::Adam optimizer;
  optimizer.StepSize() = 0.1;
  optimizer.BatchSize() = 16;
  optimizer.MaxIterations() = trainData.n_cols;

  for (int i = 0; i < 10; i++)
  {
    autoencoder.Train(trainData, trainData, optimizer);
    optimizer.ResetPolicy() = false;
  }

  REQUIRE(MeanTestLoss<>(autoencoder, trainData, 16) < 0.01);
}
