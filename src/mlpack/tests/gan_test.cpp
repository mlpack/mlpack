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

BOOST_AUTO_TEST_SUITE(GANNetworkTest);


BOOST_AUTO_TEST_CASE(GanTest)
{
  arma::mat trainData(1, 100), trainLabels(1, 100);
  trainData.randn();
  trainLabels.ones();
  size_t hiddenLayerSize = 10;
  size_t outputSize = 1;
  size_t maxEpochs = 10;

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  FFN<NegativeLogLikelihood<> > model1;
  model1.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model1.Add<SigmoidLayer<> >();
  model1.Add<Linear<> >(hiddenLayerSize, outputSize);
  model1.Add<LogSoftMax<> >();
  arma::mat mean, variance;
  mean.set_size(1, 1);
  variance.set_size(1, 1);
  mean.fill(0);
  variance.fill(1);
  GaussianDistribution noise(mean, variance);
  GaussianInitialization gaussian(0, 0.1);
  RMSProp opt(0.01, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  GenerativeAdversarialNetwork<FFN<>, FFN<>, GaussianDistribution,
      GaussianInitialization> gan(
          trainData, trainLabels, gaussian, model, model1, noise, 10);
  BOOST_REQUIRE_EQUAL(1, 0);
  // gan.Train(opt);
}
BOOST_AUTO_TEST_SUITE_END();
