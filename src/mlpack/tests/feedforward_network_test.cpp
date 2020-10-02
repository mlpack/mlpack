/**
 * @file tests/feedforward_network_test.cpp
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

#include <ensmallen.hpp>

#include "catch.hpp"
#include "serialization_catch.hpp"
#include "custom_layer.hpp"

using namespace mlpack;
using namespace mlpack::ann;

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

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::mat, typename ModelType>
void CheckCopyFunction(ModelType* network1,
                       MatType& trainData,
                       MatType& trainLabels,
                       const size_t maxEpochs)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);
  FFN<> network2;
  network2 = *network1;
  delete network1;

  // Deallocating all of network1's memory, so that
  // if network2 is trying to use any of that memory.
  arma::mat predictions2;
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::mat, typename ModelType>
void CheckMoveFunction(ModelType* network1,
                       MatType& trainData,
                       MatType& trainLabels,
                       const size_t maxEpochs)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);
  FFN<> network2(std::move(*network1));
  delete network1;

  // Deallocating all of network1's memory, so that
  // if network2 is trying to use any of that memory.
  arma::mat predictions2;
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

/**
 * Check whether copying and moving Vanila network is working or not.
 */
TEST_CASE("CheckCopyMovingVanillaNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  /*
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

  FFN<NegativeLogLikelihood<> > *model = new FFN<NegativeLogLikelihood<> >;
  model->Add<Linear<> >(trainData.n_rows, 8);
  model->Add<SigmoidLayer<> >();
  model->Add<Linear<> >(8, 3);
  model->Add<LogSoftMax<> >();

  FFN<NegativeLogLikelihood<> > *model1 = new FFN<NegativeLogLikelihood<> >;
  model1->Add<Linear<> >(trainData.n_rows, 8);
  model1->Add<SigmoidLayer<> >();
  model1->Add<Linear<> >(8, 3);
  model1->Add<LogSoftMax<> >();

  // Check whether copy cpnstructor is working or not.
  CheckCopyFunction<>(model, trainData, trainLabels, 1);

  // Check whether move cpnstructor is working or not.
  CheckMoveFunction<>(model1, trainData, trainLabels, 1);
}

/**
 * Train the vanilla network on a larger dataset.
 */
TEST_CASE("FFVanillaNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  /*
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

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  TestNetwork<>(model, trainData, trainLabels, testData, testLabels, 10, 0.1);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model1;
  model1.Add<Linear<> >(dataset.n_rows, 10);
  model1.Add<SigmoidLayer<> >();
  model1.Add<Linear<> >(10, 2);
  model1.Add<LogSoftMax<> >();
  // Vanilla neural net with logistic activation function.
  TestNetwork<>(model1, dataset, labels, dataset, labels, 10, 0.2);
}

TEST_CASE("ForwardBackwardTest", "[FeedForwardNetworkTest]")
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(dataset.n_rows, 50);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(50, 10);
  model.Add<LogSoftMax<> >();

  ens::VanillaUpdate opt;
  model.ResetParameters();
  #if ENS_VERSION_MAJOR == 1
  opt.Initialize(model.Parameters().n_rows, model.Parameters().n_cols);
  #else
  ens::VanillaUpdate::Policy<arma::mat, arma::mat> optPolicy(opt,
      model.Parameters().n_rows, model.Parameters().n_cols);
  #endif
  double stepSize = 0.01;
  size_t batchSize = 10;

  size_t iteration = 0;
  bool converged = false;
  while (iteration < 100)
  {
    arma::running_stat<double> error;
    size_t batchStart = 0;
    while (batchStart < dataset.n_cols)
    {
      size_t batchEnd = std::min(batchStart + batchSize,
          (size_t) dataset.n_cols);
      arma::mat currentData = dataset.cols(batchStart, batchEnd - 1);
      arma::mat currentLabels = labels.cols(batchStart, batchEnd - 1);
      arma::mat currentResuls;
      model.Forward(currentData, currentResuls);
      arma::mat gradients;
      model.Backward(currentData, currentLabels, gradients);
      #if ENS_VERSION_MAJOR == 1
      opt.Update(model.Parameters(), stepSize, gradients);
      #else
      optPolicy.Update(model.Parameters(), stepSize, gradients);
      #endif
      batchStart = batchEnd;

      arma::mat prediction = arma::zeros<arma::mat>(1, currentResuls.n_cols);

      for (size_t i = 0; i < currentResuls.n_cols; ++i)
      {
        prediction(i) = arma::as_scalar(arma::find(
            arma::max(currentResuls.col(i)) == currentResuls.col(i), 1)) + 1;
      }

      size_t correct = arma::accu(prediction == currentLabels);
      error(1 - (double) correct / batchSize);
    }
    Log::Debug << "Current training error: " << error.mean() << std::endl;
    iteration++;
    if (error.mean() < 0.05)
    {
      converged = true;
      break;
    }
  }

  REQUIRE(converged);
}

/**
 * Train the dropout network on a larger dataset.
 */
TEST_CASE("DropoutNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Dropout      Output
   *  Layer         Layer         Layer        Layer
   * +-----+       +-----+       +-----+       +-----+
   * |     |       |     |       |     |       |     |
   * |     +------>|     +------>|     +------>|     |
   * |     |     +>|     |       |     |       |     |
   * +-----+     | +--+--+       +-----+       +-----+
   *             |
   *  Bias       |
   *  Layer      |
   * +-----+     |
   * |     |     |
   * |     +-----+
   * |     |
   * +-----+
   */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  TestNetwork<>(model, trainData, trainLabels, testData, testLabels, 10, 0.1);
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    dataset.col(i) /= norm(dataset.col(i), 2);
  }

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model1;
  model1.Add<Linear<> >(dataset.n_rows, 10);
  model1.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model1.Add<Linear<> >(10, 2);
  model1.Add<LogSoftMax<> >();
  // Vanilla neural net with logistic activation function.
  TestNetwork<>(model1, dataset, labels, dataset, labels, 10, 0.2);
}

