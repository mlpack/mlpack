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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

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
  VisibleLayer<> visible(trainData.n_rows, hiddenLayerSize);
  VisibleLayer<> hidden(hiddenLayerSize, trainData.n_rows);
  GaussianInitialization gaussian(0,1);
  VanillaRBM<GaussianInitialization, VisibleLayer<>, VisibleLayer<> > model(gaussian,visible, hidden);

  // Test the Sample function
  arma::mat output;
  model.Reset();
  arma::mat temp = trainData.cols(0, 0);
  model.SampleHidden(std::move(temp), std::move(output));
  model.SampleVisible(std::move(output), std::move(output));
  
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  arma::mat dataset;
  dataset.load("r10.txt");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  // Vanilla neural net with logistic activation function.
  BuildVanillaNetwork<>
      (dataset,10);

  //std::cout << "here" << std::endl;

}
BOOST_AUTO_TEST_SUITE_END();