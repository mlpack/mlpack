#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/rbm.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(RbmNetworkTest);

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
  RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<>> model(trainData,
      gaussian, visible, hidden, 1,  true);
  MiniBatchSGD msgd(10,0.06, trainData.n_cols * 20, 0, true);
  model.Train(trainData, msgd);
  model.Reset();
  // Set the parmaeters from a learned rbm
  model.Parameters() = {-0.23224054, -0.23000632,
                        -0.25701271, -0.25122418,
                        -0.20716651,  -0.20962217,
                        -0.59922456, -0.60003836,
                        -0.6, -0.625, -0.475};
  // Check Weight Shared
  BOOST_REQUIRE_CLOSE(arma::accu(model.VisibleLayer().Weight() - 
      model.HiddenLayer().Weight()), 0, 1e-2);

  // Check free energy
  arma::vec freeEnergy = {-0.87523715, 0.50615066, 0.46923476, 1.21509084};
  arma::vec calcultedFreeEnergy(4);
  calcultedFreeEnergy.zeros();
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    model.VisibleLayer().ForwardPreActivation(std::move(trainData.col(i)),
        std::move(output));
    calcultedFreeEnergy(i) = model.FreeEnergy(std::move(trainData.col(i)));
  }

  BOOST_REQUIRE_LE(arma::accu(calcultedFreeEnergy - freeEnergy), 1);
  
}
BOOST_AUTO_TEST_CASE(MiscTest)
{
  /**
   * Train and evaluate a vanilla network with the specified structure.
   */

  arma::mat X;
  X = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
  X = X.t();
  BuildVanillaNetwork<>(X, 2);
}

BOOST_AUTO_TEST_CASE(ClassificationTest)
{
  // Normalised dataset
  int hiddenLayerSize = 100;
  arma::mat trainData, testData, dataset;
  arma::vec trainLabelsTemp, testLabelsTemp;
  data::Load("mnist_data_small.csv", dataset, true);
  trainData = dataset.submat(0, 0,
      dataset.n_rows - 102, dataset.n_cols - 2);
  testData = dataset.submat(dataset.n_rows - 102, 0 ,
      dataset.n_rows - 2, dataset.n_cols - 2);
  trainLabelsTemp = dataset.submat(0, dataset.n_cols - 1,
      dataset.n_rows - 102, dataset.n_cols - 1);
  testLabelsTemp = dataset.submat(dataset.n_rows - 102, dataset.n_cols - 1,
      dataset.n_rows - 2, dataset.n_cols - 1);

  arma::Row<size_t> trainLabels = arma::zeros<arma::Row<size_t>>(1,
      trainLabelsTemp.n_rows);
  arma::Row<size_t> testLabels = arma::zeros<arma::Row<size_t>>(1,
      testLabelsTemp.n_rows);
  
  for (size_t i = 0; i < trainLabelsTemp.n_rows; ++i)
  {
    trainLabels(i) = arma::as_scalar(trainLabelsTemp.row(i));
  }

  for (size_t i = 0; i < testLabels.n_rows; ++i)
  {
    testLabels(i) = arma::as_scalar(testLabels.row(i));
  }
  trainData = trainData.t();
  testData = testData.t();

  arma::mat output, XRbm(hiddenLayerSize, trainData.n_cols), 
      YRbm(hiddenLayerSize, trainLabels.n_cols);

  XRbm.zeros();
  YRbm.zeros();
  
  BinaryLayer<> visible(trainData.n_rows, hiddenLayerSize, 1);
  BinaryLayer<> hidden(hiddenLayerSize, trainData.n_rows, 0);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> > model(trainData,
      gaussian, visible, hidden, 1,  true, true);
  MiniBatchSGD msgd(10,0.06, trainData.n_cols * 20, 0, true);
  model.Reset();
  model.VisibleLayer().Bias().ones();
  model.HiddenLayer().Bias().ones();
  // test the reset function
  model.Train(trainData, msgd);

  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    model.VisibleLayer().Forward(std::move(trainData.col(i)), 
        std::move(output));
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < testData.n_cols; i++)
  {
    model.VisibleLayer().Forward(std::move(testData.col(i)), std::move(output));
    YRbm.col(i) = output;
  }
  const size_t numClasses = 10; // Number of classes.
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 100; // Maximum number of iterations.
  
  // Use an instantiated optimizer for the training.
  L_BFGS optimizer(numBasis, numIterations);
  SoftmaxRegression regressor2(trainData, trainLabels,
      numClasses, 0.001, false, optimizer);
 
  arma::Row<size_t> predictions1, predictions2;
  // Vectors to store predictions in.

  double classificationAccuray = regressor2.ComputeAccuracy(testData,
   testLabels);
  std::cout << classificationAccuray << std::endl;
  
  L_BFGS optimizer1(numBasis, numIterations);
  SoftmaxRegression regressor1(XRbm, trainLabels, numClasses,
        0.001, false, optimizer1);
  double classificationAccuray1 = regressor1.ComputeAccuracy(YRbm, testLabels);
  std::cout << classificationAccuray1 << std::endl;
  BOOST_REQUIRE_GE(classificationAccuray1, classificationAccuray);
}

BOOST_AUTO_TEST_SUITE_END();
