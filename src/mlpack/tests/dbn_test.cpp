#include <mlpack/core.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/dbn.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <ensmallen.hpp>


#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(DBNetworkTest);

/*
 * Tests the Single Layer DBN implementation on the Digits dataset.
 */
BOOST_AUTO_TEST_CASE(SingleLayerDBNClassificationTest)
{
  // Normalised dataset.
  int hiddenLayerSize = 256;
  size_t batchSize = 10;
  size_t numEpoches = 30;

  arma::mat trainData, testData, dataset;
  arma::mat trainLabelsTemp, testLabelsTemp;
  trainData.load("digits_train.arm");
  testData.load("digits_test.arm");
  trainLabelsTemp.load("digits_train_label.arm");
  testLabelsTemp.load("digits_test_label.arm");

  arma::Row<size_t> trainLabels = arma::zeros<arma::Row<size_t>>(1,
      trainLabelsTemp.n_cols);
  arma::Row<size_t> testLabels = arma::zeros<arma::Row<size_t>>(1,
      testLabelsTemp.n_cols);

  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
    trainLabels(i) = arma::as_scalar(trainLabelsTemp.col(i));

  for (size_t i = 0; i < testLabelsTemp.n_cols; ++i)
    testLabels(i) = arma::as_scalar(testLabelsTemp.col(i));

  arma::mat output, XRbm(hiddenLayerSize, trainData.n_cols),
      YRbm(hiddenLayerSize, testLabels.n_cols);

  XRbm.zeros();
  YRbm.zeros();

  GaussianInitialization gaussian(0, 0.1);

  // Constructing a DBN with single RBM layer.
  DBN<arma::mat, arma::mat, GaussianInitialization> model(trainData);
  model.Add(RBM<GaussianInitialization> (gaussian,
                      trainData.n_rows, hiddenLayerSize, batchSize));
  model.Reset();
  model.SetBias();

  size_t numRBMIterations = trainData.n_cols * numEpoches;
  numRBMIterations /= batchSize;
  ens::StandardSGD msgd(0.03, batchSize, numRBMIterations, 0, true);

  // Test the reset function.
  double objVal = model.Train(msgd);

  // Test that objective value returned by DBN::Train() is finite.
  BOOST_REQUIRE_EQUAL(std::isfinite(objVal), true);

  model.Forward(trainData, output);
  XRbm = output;

  model.Forward(testData, output);
  YRbm = output;

  const size_t numClasses = 10; // Number of classes.
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 100; // Maximum number of iterations.

  // Use an instantiated optimizer for the training.
  L_BFGS optimizer(numBasis, numIterations);

  SoftmaxRegression regressor(trainData, trainLabels,
      numClasses, 0.001, false, optimizer);

  double classificationAccuracy = regressor.ComputeAccuracy(testData,
    testLabels);

  L_BFGS rbmOptimizer(numBasis, numIterations);

  SoftmaxRegression rbmRegressor(XRbm, trainLabels, numClasses,
        0.001, false, rbmOptimizer);
  double rbmClassificationAccuracy = rbmRegressor.ComputeAccuracy(YRbm,
      testLabels);

  // We allow a 6% tolerance because the RBM may not reconstruct samples as
  // well.  (Typically it does, but we have no guarantee.)
  BOOST_REQUIRE_GE(rbmClassificationAccuracy, classificationAccuracy - 6.0);
}

BOOST_AUTO_TEST_SUITE_END();
