/*
 * @file ksinit_test.cpp
 * @author Praveen Ch
 *
 * Tests the working of Kathirvalavakumar Subavathi Initialization for a
 * Feed forward neural network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
*/
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(KSInitialization);

/**
 * Train and evaluate a vanilla network with the specified initialisation
 * procedure.
 *
 * @param trainData Training data.
 * @param trainLabels Training labels.
 * @param testData Test data.
 * @param testLabels Test labels.
 * @param testLabels Test labels.
 * @param hiddenLayerSize Hidden layer size.
 * @param maxEpochs Maximum number of epochs.
 * @param outputSize The size of the output layer.
 * @param trainError Mean squared error of predictions on training data.
 * @param testError Mean squared error of predictions on test data.
*/
template<typename MatType = arma::mat>
void BuildVanillaNetwork(MatType& trainData,
                         const MatType& trainLabels,
                         MatType& testData,
                         const MatType& testLabels,
                         const size_t hiddenLayerSize,
                         const size_t maxEpochs,
                         const size_t outputSize,
                         double& trainError,
                         double& testError)
{
  /**
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Output
   *  Layer         Layer         Layer
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |     +>|     |     +>|     |
   * +-----+     | +--+--+     | +-----+
   *             |             |
   *  Bias       |  Bias       |
   *  Layer      |  Layer      |
   * +-----+     | +-----+     |
   * |     |     | |     |     |
   * |     +-----+ |     +-----+
   * |     |       |     |
   * +-----+       +-----+
   */

  // 4.59 is a constant used in the "A New Weight Initialization Method Using
  // Cauchy’s Inequality Based on Sensitivity Analysis" paper.
  KathirvalavakumarSubavathiInitialization init(trainData, 4.59);

  FFN<MeanSquaredError<>, KathirvalavakumarSubavathiInitialization>
      model(MeanSquaredError<>(), init);

  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<LeakyReLU<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);

  RMSProp opt(0.01, 1, 0.88, 1e-8, maxEpochs * trainData.n_cols, 1e-18);

  model.Train(trainData, trainLabels, opt);

  MatType prediction;

  // Calculating the mean squared error on the training data.
  model.Predict(trainData, prediction);
  trainError = arma::mean(arma::mean(arma::square(prediction - trainLabels)));

  // Calculating the mean squared error on the test data
  model.Predict(testData, prediction);
  testError = arma::mean(arma::mean(arma::square(prediction - testLabels)));
}

/**
 * CrossValidation function runs a k-fold cross validation on the training data
 * by dividing the training data into k equal disjoint subsets. The model is
 * trained on k-1 of these subsets and 1 subset is used as validation data.
 *
 * This process is repeated k times assigning each subset to be the validation
 * data at most once.
 *
 * @params trainData The data available for training.
 * @params trainLabels The labels corresponding to the training data.
 * @params k The parameter k in k-fold cross validation.
 * @param hiddenLayerSize Hidden layer size.
 * @param maxEpochs Maximum number of epochs.
 * @param trainError Error of predictions on training data.
 * @param validationError Validation error of predictions.
*/
void CrossValidation(arma::mat& trainData,
                     const arma::mat& trainLabels,
                     const size_t k,
                     const size_t hiddenLayerSize,
                     const size_t maxEpochs,
                     double& trainError,
                     double& validationError)
{
  // Number of datapoints in each subset in K-fold CV.
  size_t validationDataSize = (int) trainData.n_cols / k;
  trainError = validationError = 0.0;

  for (size_t i = 0; i < trainData.n_cols; i = i + validationDataSize)
  {
    validationDataSize = (int) trainData.n_cols / k;

    // The collection of the k-1 subsets to be used in training in a particular
    // iteration.
    arma::mat validationTrainData(trainData.n_rows, trainData.n_cols);

    // The labels corresponding to training data.
    arma::mat validationTrainLabels(trainLabels.n_rows, trainLabels.n_cols);

    // The data subset which is used as validation data in a particular
    // iteration.
    arma::mat validationTestData(trainData.n_rows, validationDataSize);

    // The labels corresponding to the validation data.
    arma::mat validationTestLabels(trainLabels.n_rows, validationDataSize);

    if (i + validationDataSize > trainData.n_cols)
    {
      validationDataSize = trainData.n_cols - i;
    }

    validationTestData = trainData.submat(0, i, trainData.n_rows - 1,
        i + validationDataSize - 1);

    validationTestLabels = trainLabels.submat(0, i, trainLabels.n_rows - 1,
        i + validationDataSize - 1);

    validationTrainData = trainData;
    validationTrainData.shed_cols(i, i + validationDataSize - 1);

    validationTrainLabels = trainLabels;
    validationTrainLabels.shed_cols(i, i + validationDataSize - 1);

    double tError, vError;

    BuildVanillaNetwork(validationTrainData, validationTrainLabels,
        validationTestData, validationTestLabels, hiddenLayerSize, maxEpochs,
        validationTrainLabels.n_rows, tError, vError);

    trainError += tError;
    validationError += vError;
  }

  trainError /= k;
  validationError /= k;
}

/**
 * AvgCrossValidation function takes a dataset and runs CrossValidation "iter"
 * number of times and then return the average training and validation error.
 * It shuffles the dataset in every iteration.
 *
 * @params dataset The dataset inclusive of the labels. Assuming the last
 *                 "numLabels" number of rows are the labels which the model
 *                 has to predict.
 * @params numLabels number of rows which are the output labels in the dataset.
 * @params iter The number of times Cross Validation has to be run.
 * @params hiddenLayerSize The number of nodes in the hidden layer.
 * @params maxEpochs The maximum number of epochs for the training.
 * @param avgTrainError Average error.
 * @param validationError Average validation.
 */
void AvgCrossValidation(arma::mat& dataset,
                        const size_t numLabels,
                        const size_t iter,
                        const size_t hiddenLayerSize,
                        const size_t maxEpochs,
                        double& avgTrainError,
                        double& avgValidationError)
{
  avgValidationError = avgTrainError = 0.0;

  for (size_t i = 0; i < iter; ++i)
  {
    dataset = arma::shuffle(dataset, 1);

    arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 1 - numLabels,
        dataset.n_cols - 1);
    arma::mat trainLabels = dataset.submat(dataset.n_rows - numLabels, 0,
        dataset.n_rows - 1, dataset.n_cols - 1);

    double trainError, validationError;
    CrossValidation(trainData, trainLabels, 10, hiddenLayerSize, maxEpochs,
        trainError, validationError);

    avgTrainError += trainError;
    avgValidationError += validationError;
  }

  avgTrainError /= iter;
  avgValidationError /= iter;
}

/*
 * Kathirvalavakumar Subavathi Initialization test case for the Iris Dataset.
 */
BOOST_AUTO_TEST_CASE(IrisDataset)
{
  double trainErrorThreshold = 0.01;
  double validationErrorThreshold = 0.01;

  arma::mat dataset, labels;

  data::Load("iris.csv", dataset, true);
  data::Load("iris_labels.txt", labels, true);

  dataset.insert_rows(dataset.n_rows, labels);

  // Normalization used in the paper.
  dataset /= 10;

  // Counter for the number of failures.
  size_t numFails = 0;

  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using partially random weights. If this works 1 of 5 times,
  // I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  while (numFails < 5)
  {
    double avgTrainError, avgValidationError;

    // Run the CV for 10 times.
    AvgCrossValidation(dataset, 1, 10, 3, 15, avgTrainError,
        avgValidationError);

    if (avgTrainError <= trainErrorThreshold &&
        avgValidationError <= validationErrorThreshold)
    {
      break;
    }

    ++numFails;
  }

  BOOST_REQUIRE_LE(numFails, 4);
}

/*
 * Kathirvalavakumar Subavathi Initialization Test case for
 * the Non Linear Function Approximation Problem.
 */
BOOST_AUTO_TEST_CASE(NonLinearFunctionApproximation)
{
  double trainErrorThreshold = 0.0045;
  double validationErrorThreshold = 0.0045;

  arma::mat dataset(11, 500, arma::fill::randu);

  // Eqn 13.1 as given in the paper.
  dataset.row(8) = dataset.row(0) % dataset.row(1);

  for (size_t i = 2; i <= 6; i += 2)
  {
    dataset.row(8) += dataset.row(i) % dataset.row(i + 1);
  }

  dataset.row(8) /= 4;

  // Eqn 13.2.
  dataset.row(9) = dataset.row(0);

  for (size_t i = 1; i <= 7; ++i)
  {
    dataset.row(9) += dataset.row(i);
  }

  dataset.row(9) /= 8;

  // Eqn 13.3.
  dataset.row(10) = arma::sqrt(1 - dataset.row(0));

  // Counter for the number of failures.
  size_t numFails = 0;

  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using partially random weights. If this works 1 of 5 times,
  // I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  while (numFails < 5)
  {
    double avgTrainError, avgValidationError;

    // Run CV 5 times.
    AvgCrossValidation(dataset, 3, 5, 10, 10, avgTrainError,
        avgValidationError);

    if (avgTrainError <= trainErrorThreshold &&
        avgValidationError <= validationErrorThreshold)
    {
      break;
    }

    ++numFails;
  }

  BOOST_REQUIRE_LE(numFails, 4);
}

BOOST_AUTO_TEST_SUITE_END();
