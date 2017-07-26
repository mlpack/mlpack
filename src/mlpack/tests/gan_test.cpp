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

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/gan.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

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
  size_t hiddenLayerSize1 = 500;
  size_t hiddenLayerSize2 = 500;
  size_t gOutputSize = 784;
  size_t gInputSize = 100;
  size_t dOutputSize = 1;
  size_t batchSize = 100;
  // Load the dataset
  arma::mat trainData, dataset;
  trainData.load("train4.txt");
  std::cout << arma::size(trainData) << std::endl;
  // Discriminator network
  FFN<CrossEntropyError<> > discriminator;
  discriminator.Add<Linear<> >(gOutputSize, hiddenLayerSize1);
  discriminator.Add<ReLULayer<>>();
  discriminator.Add<Linear<> >(hiddenLayerSize1, hiddenLayerSize2);
  discriminator.Add<ReLULayer<>>();
  discriminator.Add<Linear<> >(hiddenLayerSize1, dOutputSize);
  discriminator.Add<SigmoidLayer<> >();

  // Generator network
  FFN<CrossEntropyError<>> generator;
  generator.Add<Linear<> >(gInputSize, hiddenLayerSize1);
  generator.Add<ReLULayer<> >();
  generator.Add<Linear<> >(hiddenLayerSize1, hiddenLayerSize2);
  generator.Add<ReLULayer<> >();
  generator.Add<Linear<> >(hiddenLayerSize2, gOutputSize);
  generator.Add<SigmoidLayer<> >();

  // Intialisation function
  GaussianInitialization gaussian(0, 0.1);
  // Optimizer
  MiniBatchSGD msgd(batchSize, 0.05, trainData.n_cols, 0.001, true);
  // GAN model
  GenerativeAdversarialNetwork<FFN<CrossEntropyError<>>, FFN<CrossEntropyError<>>,
      GaussianInitialization> gan(
      trainData, gaussian, generator, discriminator, batchSize, 100, 10,
      gInputSize);
  gan.Train(msgd);
}
BOOST_AUTO_TEST_SUITE_END();
