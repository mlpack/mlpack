/**
 * @file tests/feedforward_network_2_test.cpp
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
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

#include <ensmallen.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "custom_layer.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::kmeans;

/**
 * Train and evaluate a model with the specified structure.
 */
template<typename MatType = arma::mat, typename ModelType>
void TestNetwork(ModelType& model,
                 MatType& trainData,
                 MatType& trainLabels,
                 MatType& testData,
                 MatType& testLabels,
                 const size_t maxEpochs,
                 const double classificationErrorThreshold)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  size_t correct = arma::accu(prediction == testLabels);
  double classificationError = 1 - double(correct) / testData.n_cols;
  REQUIRE(classificationError <= classificationErrorThreshold);
}

/**
 * Train the RBF network on a larger dataset.
 */
TEST_CASE("RBFNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat trainLabels1 = arma::zeros(3, trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    trainLabels1.col(i).row((trainLabels(i) - 1)) = 1;
  }

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         RBF          Activation    Output
   *  Layer         Layer         Layer        Layer
   * +-----+       +-----+       +-----+       +-----+
   * |     |       |     |       |     |       |     |
   * |     +------>|     +------>|     +------>|     |
   * |     |       |     |       |     |       |     |
   * +-----+       +--+--+       +-----+       +-----+
   */
  arma::mat centroids;
  KMeans<> kmeans;
  kmeans.Cluster(trainData, 8, centroids);

  FFN<MeanSquaredError<> > model;
  model.Add<RBF<> >(trainData.n_rows, 8, centroids);
  model.Add<Linear<> >(8, 3);

  // RBFN neural net with MeanSquaredError.
  TestNetwork<>(model, trainData, trainLabels1, testData, testLabels, 10, 0.1);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    dataset.col(i) /= norm(dataset.col(i), 2);
  }

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  arma::mat labels1 = arma::zeros(2, dataset.n_cols);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    labels1.col(i).row(labels(i)) = 1;
  }
  labels += 1;

  arma::mat centroids1;
  arma::Row<size_t> assignments;
  KMeans<> kmeans1;
  kmeans1.Cluster(dataset, 140, centroids1);

  FFN<MeanSquaredError<> > model1;
  model1.Add<RBF<> >(dataset.n_rows, 140, centroids1, 4.1);
  model1.Add<Linear<> >(140, 2);

  // RBFN neural net with MeanSquaredError.
  TestNetwork<>(model1, dataset, labels1, dataset, labels, 10, 0.1);
}
