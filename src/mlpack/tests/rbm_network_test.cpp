/**
 * @file rbm_network_test.cpp
 * @author Marcus Edel
 * @author Palash Ahuja
 *
 * Tests the feed forward network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/vanilla_rbm.hpp>
#include <mlpack/core/optimizers/cdk/cdk.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(RBMNetworkTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<typename MatType = arma::mat>
void BuildVanillaNetwork(MatType& trainData,
                         const size_t hiddenLayerSize)
{
  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Visible       Hidden        
   *  Layer         Layer         
   * +-----+       +-----+       
   * |     |       |     |            
   * |     +<----->|     |
   * |     |       |     | 
   * +-----+       +-----+     
   *        
   */
  arma::mat output;
  BinaryLayer<> visible(trainData.n_rows, hiddenLayerSize, 1);
  BinaryLayer<> hidden(hiddenLayerSize, trainData.n_rows, 0);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> > model(trainData,
      gaussian, visible, hidden);
  CDK<RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> >> cdk(model,
      10, 0.1, 1000, 20, true, true);
  model.Reset();
  model.VisibleLayer().Bias().zeros();
  model.HiddenLayer().Bias().zeros();
  // test the reset function
  model.Train(trainData, cdk);
  model.SampleHidden(std::move(trainData.col(46)), std::move(output));
  std::cout << output << std::endl;
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  arma::mat dataset;
  dataset.load("/Users/kris/Desktop/GsoC/mlpack/src/mlpack/tests/data/mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);
  BuildVanillaNetwork<>(dataset, 500);
}
BOOST_AUTO_TEST_SUITE_END();
