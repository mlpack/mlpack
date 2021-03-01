/**
 * @file tests/rbm_network_test.cpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * Tests the RBM Network
 *
 * digits dataset source:
 * @misc{Lichman:2013 ,
 * author = "M. Lichman",
 * year = "2013",
 * title = "{UCI} Machine Learning Repository",
 * url = "http://archive.ics.uci.edu/ml",
 * institution = "University of California,
 * Irvine, School of Information and Computer Sciences" }
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/rbm/rbm.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <ensmallen.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::regression;

/*
 * Tests the BinaryRBM implementation on the Digits dataset.
 */
TEST_CASE("BinaryRBMClassificationTest", "[RBMNetworkTest]")
{
  // Normalised dataset.
  int hiddenLayerSize = 100;
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
  RBM<GaussianInitialization> model(trainData,
      gaussian, trainData.n_rows, hiddenLayerSize, batchSize);

  size_t numRBMIterations = trainData.n_cols * numEpoches;
  numRBMIterations /= batchSize;
  ens::StandardSGD msgd(0.03, batchSize, numRBMIterations, 0, true);
  model.Reset();
  model.VisibleBias().ones();
  model.HiddenBias().ones();

  // Test the reset function.
  double objVal = model.Train(msgd);

  // Test that objective value returned by RBM::Train() is finite.
  REQUIRE(std::isfinite(objVal) == true);

  for (size_t i = 0; i < trainData.n_cols; ++i)
  {
    model.HiddenMean(trainData.col(i), output);
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < testData.n_cols; ++i)
  {
    model.HiddenMean(testData.col(i), output);
    YRbm.col(i) = output;
  }
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
  REQUIRE(rbmClassificationAccuracy >= classificationAccuracy - 6.0);
}

/*
 * Tests the SpikeSlabRBM implementation on the Digits dataset.
 */
TEST_CASE("ssRBMClassificationTest", "[RBMNetworkTest]")
{
  size_t batchSize = 10;
  size_t numEpoches = 3;
  int hiddenLayerSize = 80;
  double radius = 0;
  double tempRadius = 0;
  arma::mat trainData, testData, dataset;
  arma::mat trainLabelsTemp, testLabelsTemp;
  trainData.load("digits_train.arm");
  testData.load("digits_test.arm");
  trainLabelsTemp.load("digits_train_label.arm");
  testLabelsTemp.load("digits_test_label.arm");
  GaussianInitialization gaussian(0, 1);

  arma::Row<size_t> trainLabels = arma::zeros<arma::Row<size_t>>(1,
      trainLabelsTemp.n_cols);
  arma::Row<size_t> testLabels = arma::zeros<arma::Row<size_t>>(1,
      testLabelsTemp.n_cols);

  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
    trainLabels(i) = arma::as_scalar(trainLabelsTemp.col(i));

  for (size_t i = 0; i < testLabelsTemp.n_cols; ++i)
    testLabels(i) = arma::as_scalar(testLabelsTemp.col(i));

  for (size_t i = 0; i < trainData.n_cols; ++i)
  {
    tempRadius = arma::norm(trainData.col(i));
    if (radius < tempRadius)
      radius = tempRadius;
  }

  size_t poolSize = 2;
  radius *= 1.3;

  arma::mat output;
  arma::mat XRbm(hiddenLayerSize + poolSize * hiddenLayerSize,
      trainData.n_cols);
  arma::mat YRbm(hiddenLayerSize + poolSize * hiddenLayerSize,
      testLabels.n_cols);

  XRbm.zeros();
  YRbm.zeros();
  double slabPenalty = 8;

  RBM<GaussianInitialization, arma::mat, SpikeSlabRBM> modelssRBM(trainData,
      gaussian, trainData.n_rows, hiddenLayerSize, batchSize, 1, 1, poolSize,
      slabPenalty, radius);

  size_t numRBMIterations = trainData.n_cols * numEpoches;
  numRBMIterations /= batchSize;

  ens::StandardSGD msgd(0.02, batchSize, numRBMIterations, 0, true);
  modelssRBM.Reset();
  modelssRBM.VisiblePenalty().fill(5);
  modelssRBM.SpikeBias().fill(1);

  double objVal = modelssRBM.Train(msgd);

  // Test that objective value returned by RBM::Train() is finite.
  REQUIRE(std::isfinite(objVal) == true);

  for (size_t i = 0; i < trainData.n_cols; ++i)
  {
    modelssRBM.HiddenMean(trainData.col(i), output);
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < testData.n_cols; ++i)
  {
    modelssRBM.HiddenMean(testData.col(i), output);
    YRbm.col(i) = output;
  }
  const size_t numClasses = 10; // Number of classes.
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 100; // Maximum number of iterations.

  L_BFGS ssRbmOptimizer(numBasis, numIterations);
  SoftmaxRegression ssRbmRegressor(XRbm, trainLabels, numClasses,
        0.001, false, ssRbmOptimizer);
  double ssRbmClassificationAccuracy = ssRbmRegressor.ComputeAccuracy(
      YRbm, testLabels);

  // 76.18 is the standard accuracy of the Softmax regression classifier,
  // omitted here for speed.  We add a margin of 3% since ssRBM isn't guaranteed
  // to give us better results (we just generally expect it to be about as good
  // or better).
  REQUIRE(ssRbmClassificationAccuracy >= 76.18 - 3.0);
}

template<typename MatType = arma::mat>
void BuildVanillaNetwork(MatType& trainData,
                         const size_t hiddenLayerSize)
{
  MatType output;
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, MatType, BinaryRBM> model(trainData, gaussian,
      trainData.n_rows, hiddenLayerSize, 1, 1, 1, 2, 8, 1, true);

  model.Reset();
  // Set the parameters from a learned RBM Sklearn random state 23.
  model.Parameters() = MatType(
      "-0.23224054, -0.23000632, -0.25701271, -0.25122418, -0.20716651,"
      "-0.20962217, -0.59922456, -0.60003836, -0.6, -0.625, -0.475;");

  // Check free energy.
  arma::Mat<float> freeEnergy = MatType(
      "-0.87523715, 0.50615066, 0.46923476, 1.21509084;");
  arma::vec calculatedFreeEnergy(4, arma::fill::zeros);
  for (size_t i = 0; i < trainData.n_cols; ++i)
  {
    calculatedFreeEnergy(i) = model.FreeEnergy(trainData.col(i));
  }

  for (size_t i = 0; i < freeEnergy.n_elem; ++i)
    REQUIRE(calculatedFreeEnergy(i) == Approx(freeEnergy(i)).epsilon(1e-5));
}

/*
 * Train and evaluate a Vanilla network with the specified structure.
 */
TEST_CASE("MiscTest", "[RBMNetworkTest]")
{
  arma::Mat<float> X = arma::Mat<float>("0.0, 0.0, 0.0;"
                          "0.0, 1.0, 1.0;"
                          "1.0, 0.0, 1.0;"
                          "1.0, 1.0, 1.0;");
  X = X.t();
  BuildVanillaNetwork<arma::Mat<float>>(X, 2);
}
