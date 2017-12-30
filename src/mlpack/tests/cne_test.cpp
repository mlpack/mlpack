/**
 * @file cne_test.cpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Test file for CNE (Conventional Neural Evolution).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <mlpack/core/optimizers/cne/cne.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

using namespace mlpack::distribution;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(CNETest);

/**
 * Training a vanilla network for 2 input XOR function
 */
BOOST_AUTO_TEST_CASE(CNEXORTest)
{
  /*
   * Create the four cases for XOR with two variable
   *
   *  Input    Output
   * 0 XOR 0  =  0
   * 1 XOR 1  =  0
   * 0 XOR 1  =  1
   * 1 XOR 0  =  1
   */
  arma::mat train("1, 0, 0, 1; 1, 0, 1, 0");
  arma::mat labels("1, 1, 2, 2");

  // CNE may fail to find a good optimum.  But if it can succeed one out of 6
  // times I think that is sufficient to say it is working.
  size_t successes = 0;
  for (size_t trial = 0; trial < 6; ++trial)
  {
    // Build a network with 2 input, 2 hidden, and 2 output layers.
    FFN<NegativeLogLikelihood<> > network;

    network.Add<Linear<> >(2, 2);
    network.Add<SigmoidLayer<> >();
    network.Add<Linear<> >(2, 2);
    network.Add<LogSoftMax<> >();

    // CNE object.
    CNE opt(60, 5000, 0.1, 0.02, 0.2, 0.1, -1);

    // Training the network with CNE
    network.Train(train, labels, opt);

    // Predicting for the same train data
    arma::mat predictionTemp;
    network.Predict(train, predictionTemp);

    arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

    for (size_t i = 0; i < predictionTemp.n_cols; ++i)
    {
      prediction(i) = arma::as_scalar(arma::find(
          arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
    }

    // 1 means 0 and 2 means 1 as the output to XOR.
    if ((prediction[0] == 1) &&
        (prediction[1] == 1) &&
        (prediction[2] == 2) &&
        (prediction[3] == 2))
    {
      ++successes;
      break;
    }
  }

  BOOST_REQUIRE_GT(successes, 0);
}

/**
 * Train and test a logistic regression function using CNE optimizer
 */
BOOST_AUTO_TEST_CASE(CNELogisticRegressionTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  CNE opt(200, 10000, 0.2, 0.2, 0.3, 65, -1);

  LogisticRegression<> lr(shuffledData, shuffledResponses, opt, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Training a vanilla network on a larger dataset using CNE optimizer.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkWithCNETest)
{
  // Load the datasets.
  arma::mat trainData;
  data::Load("iris_train.csv", trainData, true);

  arma::mat testData;
  data::Load("iris_test.csv", testData, true);

  arma::mat trainLabels;
  data::Load("iris_train_labels.csv", trainLabels, true);
  trainLabels += 1;

  arma::mat testLabels;
  data::Load("iris_test_labels.csv", testLabels, true);
  testLabels += 1;

  // Training the network may fail, so we will try a few times.
  size_t successes = 0;
  for (size_t trial = 0; trial < 4; ++trial)
  {
    // Create vanilla network with 4 input, 4 hidden and 3 output nodes.
    FFN<NegativeLogLikelihood<> > model;
    model.Add<Linear<> >(trainData.n_rows, 4);
    model.Add<SigmoidLayer<> >();
    model.Add<Linear<> >(4, 3);
    model.Add<LogSoftMax<> >();

    // Creating CNE object.
    // The tolerance and objectiveChange are not taken into consideration.
    CNE opt(30, 200, 0.2, 0.2, 0.3, -1, -1);

    model.Train(trainData, trainLabels, opt);

    arma::mat predictionTemp;
    model.Predict(testData, predictionTemp);
    arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

    for (size_t i = 0; i < predictionTemp.n_cols; ++i)
    {
      prediction(i) = arma::as_scalar(arma::find(
          arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
    }

    size_t error = 0;
    for (size_t i = 0; i < testData.n_cols; i++)
    {
      if (int(arma::as_scalar(prediction.col(i))) ==
          int(arma::as_scalar(testLabels.col(i))))
      {
        error++;
      }
    }

    double classificationError = 1 - double(error) / testData.n_cols;
    if (classificationError <= 0.1)
    {
      ++successes;
      break;
    }
  }

  BOOST_REQUIRE_GT(successes, 0);
}

BOOST_AUTO_TEST_SUITE_END();
