/**
 * @file rbm_network_test.cpp
 *
 * Tests the feed forward network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/cdk/cdk.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/vanilla_rbm.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(RBMNetworkTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<typename MatType = arma::mat>
arma::mat BuildVanillaNetwork(MatType& trainData,
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
  arma::mat output, datasetRBM(trainData.n_cols, hiddenLayerSize);
  BinaryLayer<> visible(trainData.n_rows, hiddenLayerSize, 1);
  BinaryLayer<> hidden(hiddenLayerSize, trainData.n_rows, 0);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> > model(trainData,
      gaussian, visible, hidden, 1, true);
  CDK<RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> >> cdk(model,
      0.06, trainData.n_cols * 20, 10, true);
  model.Reset();
  model.VisibleLayer().Bias().ones();
  model.HiddenLayer().Bias().ones();
  // test the reset function
  model.Train(trainData, cdk);

  for (size_t i = 0; i < trainData.n_cols; i++)
    model.SampleHidden(std::move(trainData.col(i)), std::move(datasetRBM.col(i)));
  return datasetRBM;
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  // Normalised dataset
  arma::mat X, Y;
  arma::Row<size_t> Xlabel, Ylabel;
  data::Load("mnist_regression.txt", X, true);
  data::Load("mnist_regression_label.txt", Xlabel, true);
  data::Load("mnist_regression_test.txt", Y, true);
  data::Load("mnist_regression_test_label.txt", Ylabel, true);
  

  arma::mat XRbm = BuildVanillaNetwork<>(X, 100);
  arma::mat YRbm = BuildVanillaNetwork<>(Y, 100);
  std::cout << XRbm.n_cols << std::endl;
  std::cout << XRbm.n_rows << std::endl;

   
   const size_t inputSize = 64; // Size of input feature vector.
   const size_t numClasses = 10; // Number of classes.
 
   const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
   const size_t numIterations = 100; // Maximum number of iterations.
  
   // Use an instantiated optimizer for the training.
   SoftmaxRegressionFunction srf(X, Xlabel, numClasses, 0.001);
   L_BFGS<SoftmaxRegressionFunction> optimizer(srf, numBasis, numIterations);
   SoftmaxRegression<L_BFGS> regressor2(optimizer);
 
   arma::Row<size_t> predictions1, predictions2; // Vectors to store predictions in.

   double classificationAccuray = regressor2.ComputeAccuracy(Y, Ylabel);
   std::cout << "classificationAccuray" << classificationAccuray << std::endl;


   SoftmaxRegressionFunction srf1(XRbm, Xlabel, numClasses, 0.001);
   L_BFGS<SoftmaxRegressionFunction> optimizer1(srf1, numBasis, numIterations);
   SoftmaxRegression<L_BFGS> regressor1(optimizer1);
   double classificationAccuray1 = regressor1.ComputeAccuracy(YRbm, Ylabel);  
   std::cout << "classificationAccuray1" << classificationAccuray1 << std::endl;

}
BOOST_AUTO_TEST_SUITE_END();
