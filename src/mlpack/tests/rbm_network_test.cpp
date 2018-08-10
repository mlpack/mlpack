/**
 * @file rbm_network_test.cpp
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

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/rbm/rbm.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(RBMNetworkTest);

/*
 * Tests the BinaryRBM implementation on the Digits dataset.
 */
BOOST_AUTO_TEST_CASE(BinaryRBMClassificationTest)
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
  optimization::StandardSGD msgd(0.03, batchSize, numRBMIterations, 0, true);
  model.Reset();
  model.VisibleBias().ones();
  model.HiddenBias().ones();

  // Test the reset function.
  model.Train(msgd);

  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    model.HiddenMean(std::move(trainData.col(i)), std::move(output));
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < testData.n_cols; i++)
  {
    model.HiddenMean(std::move(testData.col(i)),
      std::move(output));
    YRbm.col(i) = output;
  }
  const size_t numClasses = 10; // Number of classes.
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 100; // Maximum number of iterations.

  // Use an instantiated optimizer for the training.
  L_BFGS optimizer(numBasis, numIterations);
  SoftmaxRegression regressor(trainData, trainLabels,
      numClasses, 0.001, false, optimizer);

  double classificationAccuray = regressor.ComputeAccuracy(testData,
   testLabels);

  L_BFGS rbmOptimizer(numBasis, numIterations);
  SoftmaxRegression rbmRegressor(XRbm, trainLabels, numClasses,
        0.001, false, rbmOptimizer);
  double rbmClassificationAccuracy = rbmRegressor.ComputeAccuracy(YRbm,
      testLabels);

  BOOST_REQUIRE_GE(rbmClassificationAccuracy, classificationAccuray);
}

/*
 * Tests the SpikeSlabRBM implementation on the Digits dataset.
 */
BOOST_AUTO_TEST_CASE(SpikeSlabRBMClassificationTest)
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

  for (size_t i = 0; i < trainData.n_cols; i++)
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

  optimization::StandardSGD msgd(0.02, batchSize, numRBMIterations, 0, true);
  modelssRBM.Reset();
  modelssRBM.VisiblePenalty().fill(5);
  modelssRBM.SpikeBias().fill(1);

  modelssRBM.Train(msgd);
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    modelssRBM.HiddenMean(std::move(trainData.col(i)),
        std::move(output));
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < testData.n_cols; i++)
  {
    modelssRBM.HiddenMean(std::move(testData.col(i)),
      std::move(output));
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
  // omitted here for speed.
  BOOST_REQUIRE_GE(ssRbmClassificationAccuracy, 76.18);
}

/*
 * Tests the SpikeSlabRBM implementation on the CIFAR-10 dataset.
 */
BOOST_AUTO_TEST_CASE(SpikeSlabRBMCIFARTest)
{
  arma::mat dataset, labels;
  size_t datasetMaxCols = 500;
  size_t hiddenLayerSize = 256;
  size_t numPatches = 49 * 500;
  size_t batchSize = 10;
  double stepSize = 0.0001;
  size_t numEpoches = 10;
  double tolerance = 1e-5;
  bool shuffle = true;
  size_t poolSize = 4;
  double slabPenalty = 10.5;
  double lambdaBias = 10;

  std::string datafile = "";
  std::string labelfile = "";

  std::cout << "dataset = '" << dataset << "'" << std::endl;

  std::cout << std::boolalpha
      << "hiddenLayerSize = " << hiddenLayerSize
      << " batchSize = " << batchSize
      << " stepSize = " << stepSize
      << " numEpoches = " << numEpoches
      << " tolerance = " << tolerance
      << " shuffle = " << shuffle << std::endl;

  arma::mat tempData;
  double radius = 0;
  double tempRadius = 0;

  // Number of patches.
  size_t numImages = numPatches / 49;
  std::cout << "numImages" << numImages << std::endl;
  size_t trainSize = numImages - numImages / 10;
  size_t labelSize = numImages - numImages / 10;
  size_t testSize = numPatches - trainSize;

  dataset.load(datafile);
  labels.load(labelfile);

  std::cout << "dataset size "<< arma::size(dataset) << std::endl;
  std::cout << dataset.n_cols << std::endl;
  std::cout << "trainSize" << trainSize << std::endl;
  std::cout << "label size " << arma::size(labels) << std::endl;

  assert(labels.n_rows >= numImages);
  std::cout << "numPatches" << numPatches << std::endl;
  std::cout << "dataset.n_cols" << dataset.n_cols << std::endl;
  assert(dataset.n_cols >= numPatches);

  arma::mat tempTrainData(192, numPatches);

  arma::mat trainData = arma::mat(dataset.memptr(), 192, trainSize, false, false);
  arma::mat testData = arma::mat(dataset.memptr() + trainData.n_elem, 192, testSize,
      false, false);

  arma::mat trainLabelsTemp = arma::mat(labels.memptr(), labelSize , 1, false, false);
  arma::mat testLabelsTemp = arma::mat(labels.memptr() + trainLabelsTemp.n_elem,
      numImages - labelSize, 1, false, false);

  std::cout << trainLabelsTemp(trainLabelsTemp.n_elem - 1) << std::endl;
  std::cout << testLabelsTemp(testLabelsTemp.n_elem - 1) << std::endl;

  GaussianInitialization gaussian(0, 1);

  arma::Row<size_t> trainLabels = arma::zeros<arma::Row<size_t>>(1,
      trainLabelsTemp.n_rows);
  arma::Row<size_t> testLabels = arma::zeros<arma::Row<size_t>>(1,
      testLabelsTemp.n_rows);

  for (size_t i = 0; i < trainLabelsTemp.n_rows; ++i)
  {
    trainLabels(i) = arma::as_scalar(trainLabelsTemp.row(i));
  }

  for (size_t i = 0; i < testLabelsTemp.n_rows; ++i)
  {
    testLabelsTemp(i) = arma::as_scalar(testLabelsTemp.row(i));
  }

  // Calculate radius
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    tempRadius = arma::norm(trainData.col(i));
    if (radius < tempRadius)
      radius = tempRadius;
  }
  radius *=3;

  size_t iter =  (numEpoches * trainData.n_cols) / batchSize;
  std::cout << "Iterations: " << iter << std::endl;

  RBM<GaussianInitialization, arma::mat, SpikeSlabRBM> modelssRBM(trainData,
      gaussian, trainData.n_rows, hiddenLayerSize, batchSize, 1, 1, poolSize,
      slabPenalty, radius);

  optimization::StandardSGD msgd(stepSize, batchSize, iter, tolerance, shuffle);
  modelssRBM.Reset();
  modelssRBM.VisiblePenalty().fill(10);
  modelssRBM.SpikeBias().fill(-5);
  modelssRBM.Train(msgd);

  // hiddenLayer size * number of patches in one image
  arma::cube hiddenMeanTest(hiddenLayerSize , 49, testData.n_cols);
  arma::cube hiddenMeanTrain(hiddenLayerSize , 49, trainData.n_cols);
  arma::mat ssRBMFeaturesTest(hiddenLayerSize * 49, testData.n_cols);
  arma::mat ssRBMFeaturesTrain(hiddenLayerSize * 49, trainData.n_cols);

  for (size_t i = 0, j = 0; i < testData.n_cols; i++, j += 2)
  {
    j = j % 49;
    modelssRBM.SampleHidden(std::move(testData.col(i)),
        std::move(hiddenMeanTest.slice(i).col(j)));
  }

  for (size_t i = 0; i < hiddenMeanTest.n_slices; ++i)
  {
    ssRBMFeaturesTest.col(i) = arma::vectorise(hiddenMeanTest.slice(i), 1).t();
  }

  for (size_t i = 0, j = 0; i < trainData.n_cols; i++, j += 2)
  {
    j = j % 49;
    modelssRBM.SampleHidden(std::move(trainData.col(i)),
        std::move(hiddenMeanTrain.slice(i).col(j)));
  }
  for (size_t i = 0; i < hiddenMeanTrain.n_slices; ++i)
  {
    ssRBMFeaturesTrain.col(i) =
        arma::vectorise(hiddenMeanTrain.slice(i), 1).t();
  }

  std::cout << "ssRBMFeaturesTrain = " << arma::size(ssRBMFeaturesTrain)
      << std::endl;
  std::cout << "ssRBMFeaturesTest = " << arma::size(ssRBMFeaturesTest)
      << std::endl;

  arma::mat normalTrainFeat(trainData.n_rows * 49, trainData.n_cols);
  arma::mat normalTestFeat(trainData.n_rows * 49, testData.n_cols);

  trainData.resize(trainData.n_rows * 49, trainData.n_cols);
  testData.resize(trainData.n_rows * 49, testData.n_cols);

  const size_t numClasses = 10; // Number of classes.
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 1000; // Maximum number of iterations.

  arma::Row<size_t> predictions1, predictions2;
  L_BFGS optimizer1(numBasis, numIterations);
  SoftmaxRegression regressor2(normalTrainFeat, trainLabels, numClasses,
      0.001, false, optimizer1);

  SoftmaxRegression regressor1(ssRBMFeaturesTrain, trainLabels, numClasses,
      0.001, false, optimizer1);

  double classificationAccurayssRBM = regressor1.ComputeAccuracy(
      ssRBMFeaturesTest, testLabels);
  double classificationAccurayNormal = regressor2.ComputeAccuracy(
      normalTestFeat, testLabels);

  std::cout << "RBM Accuracy" << classificationAccurayssRBM << std::endl;
  std::cout << "Normal Accuracy" << classificationAccurayNormal << std::endl;
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
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    calculatedFreeEnergy(i) = model.FreeEnergy(std::move(trainData.col(i)));
  }

  for (size_t i = 0; i < freeEnergy.n_elem; i++)
    BOOST_REQUIRE_CLOSE(calculatedFreeEnergy(i), freeEnergy(i), 1e-3);
}

/*
 * Train and evaluate a Vanilla network with the specified structure.
 */
BOOST_AUTO_TEST_CASE(MiscTest)
{
  arma::Mat<float> X = arma::Mat<float>("0.0, 0.0, 0.0;"
                          "0.0, 1.0, 1.0;"
                          "1.0, 0.0, 1.0;"
                          "1.0, 1.0, 1.0;");
  X = X.t();
  BuildVanillaNetwork<arma::Mat<float>>(X, 2);
}

BOOST_AUTO_TEST_SUITE_END();
