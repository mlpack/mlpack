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
  RandomInitialization random(-1e-2, +1e-2);
  RBM<RandomInitialization, BinaryLayer<>, BinaryLayer<> > model(trainData,
      random, visible, hidden, 15, true);
  CDK<RBM<RandomInitialization, BinaryLayer<>, BinaryLayer<> >> cdk(model,
      0.01, 15 * trainData.n_cols);
  model.Reset();
  model.VisibleLayer().Bias().ones();
  model.HiddenLayer().Bias().ones();
  // test the reset function
  model.Train(trainData, cdk);
  model.Gibbs(std::move(trainData.col(0)), std::move(output), 200);
  std::cout << output << std::endl;
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);
  BuildVanillaNetwork<>(dataset, 500);
}
BOOST_AUTO_TEST_SUITE_END();
