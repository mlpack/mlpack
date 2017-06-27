#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/cdk/cdk.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/rbm.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

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
  RandomSeed(23);
  BinaryLayer<> visible(trainData.n_rows, hiddenLayerSize, 1);
  BinaryLayer<> hidden(hiddenLayerSize, trainData.n_rows, 0);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> > model(trainData,
      gaussian, visible, hidden, 1,  true);
  CDK<RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> >> cdk(model,
      0.06, trainData.n_cols * 20, 10, true);
  model.Reset();
  // Set the parmaeters from a learned rbm
  model.Parameters() = {-0.23224054, -0.23000632,
                        -0.25701271, -0.25122418,
                        -0.20716651,  -0.20962217,
                        -0.59922456, -0.60003836,
                        -0.6, -0.625, -0.475};
  // Check Weight Shared
  BOOST_REQUIRE_EQUAL(arma::accu(model.VisibleLayer().Weight() - 
      model.HiddenLayer().Weight()), 0);

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

BOOST_AUTO_TEST_CASE(ReconstructionTest)
{
  arma::mat trainData;
  arma::mat parameters;
  std::size_t hiddenLayerSize = 500;
  bool persistent = false;
  int numSteps = 1000;
  arma::mat output;

  trainData.load("mnistonly4.csv");
  parameters.load("rbm");

  trainData = trainData.t();
  for (size_t i = 0; i < trainData.n_cols; ++i)
    trainData.col(i) /= norm(trainData.col(i), 2);
  
  BinaryLayer<> visible(trainData.n_rows, hiddenLayerSize, 1);
  BinaryLayer<> hidden(hiddenLayerSize, trainData.n_rows, 0);
  RandomInitialization init(-4.0 *
      std::sqrt(6.0 / (hiddenLayerSize + trainData.n_rows)),
      4.0 * std::sqrt(6.0 / (hiddenLayerSize + trainData.n_rows)));

  RBM<RandomInitialization, BinaryLayer<>, BinaryLayer<> > model(trainData,
      init, visible, hidden, numSteps, persistent);
  model.Reset();
  model.Parameters() = parameters;

  int numSamples = 10;
  int numChains = 20;
  arma::mat mainOutput(28 * numSamples, 28 * numChains);
  arma::mat tmp;
  double result = 0;

  for (int c = 0; c < numChains; c++)
  {
    arma::mat input  = trainData.col(math::RandInt(0, trainData.n_cols));
    arma::mat copy = input;
    copy.reshape(28, 28);
    copy = copy.t();
    mainOutput.submat(0, c * 28, 27, c * 28 + 27) = copy;

    for (int s = 1; s < numSamples; s++)
    {
      for (int i = 0; i < 1000; i++)
      {
        model.SampleHidden(std::move(input), std::move(tmp));
        model.SampleVisible(std::move(tmp), std::move(input));
      }
      model.HiddenLayer().Forward(std::move(tmp), std::move(output));
      arma::mat copy = output;
      result += arma::norm(copy - input, 2);
      copy.reshape(28, 28);
      copy = copy.t();
      mainOutput.submat(s * 28, c * 28, s * 28 + 27, c * 28 + 27) = copy;
    }
  }
  BOOST_REQUIRE_LE(result / (numChains*numSamples), 700);
}

BOOST_AUTO_TEST_CASE(ClassificationTest)
{
  // Normalised dataset
  arma::mat X, Y;
  arma::Row<size_t> Xlabel, Ylabel;
  data::Load("mnist_regression.txt", X, true);
  data::Load("mnist_regression_label.txt", Xlabel, true);
  data::Load("mnist_regression_test.txt", Y, true);
  data::Load("mnist_regression_test_label.txt", Ylabel, true);
  int hiddenLayerSize = 100;

  arma::mat output, XRbm(hiddenLayerSize, X.n_cols),
      YRbm(hiddenLayerSize, Y.n_cols);

  XRbm.zeros();
  YRbm.zeros();
  
  BinaryLayer<> visible(X.n_rows, hiddenLayerSize, 1);
  BinaryLayer<> hidden(hiddenLayerSize, X.n_rows, 0);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> > model(X,
      gaussian, visible, hidden, 1,  true, true);
  CDK<RBM<GaussianInitialization, BinaryLayer<>, BinaryLayer<> >> cdk(model,
      0.06, X.n_cols * 20, 10, true);
  model.Reset();
  model.VisibleLayer().Bias().ones();
  model.HiddenLayer().Bias().ones();
  // test the reset function
  model.Train(X, cdk);

  for (size_t i = 0; i < X.n_cols; i++)
  {
    model.VisibleLayer().Forward(std::move(X.col(i)),
        std::move(output));
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < Y.n_cols; i++)
  {
    model.VisibleLayer().Forward(std::move(Y.col(i)),
        std::move(output));
    YRbm.col(i) = output;
  }

  const size_t numClasses = 10; // Number of classes.
 
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 100; // Maximum number of iterations.
  
  // Use an instantiated optimizer for the training.
  SoftmaxRegressionFunction srf(X, Xlabel, numClasses, 0.001);
  L_BFGS<SoftmaxRegressionFunction> optimizer(srf, numBasis, numIterations);
  SoftmaxRegression<L_BFGS> regressor2(optimizer);
 
  arma::Row<size_t> predictions1, predictions2;
  // Vectors to store predictions in.

  double classificationAccuray = regressor2.ComputeAccuracy(Y, Ylabel);

  SoftmaxRegressionFunction srf1(XRbm, Xlabel, numClasses, 0.001);
  L_BFGS<SoftmaxRegressionFunction> optimizer1(srf1, numBasis, numIterations);
  SoftmaxRegression<L_BFGS> regressor1(optimizer1);
  double classificationAccuray1 = regressor1.ComputeAccuracy(YRbm, Ylabel);

  BOOST_REQUIRE_GE(classificationAccuray1, 0.80);
}
BOOST_AUTO_TEST_SUITE_END();
