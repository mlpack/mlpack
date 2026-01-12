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
#ifndef MLPACK_ENABLE_ANN_SERIALIZATION
  #define MLPACK_ENABLE_ANN_SERIALIZATION
#endif
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

#include "../catch.hpp"
#include "../serialization.hpp"
#include "ann_test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;

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
  #if ENS_VERSION_MAJOR >= 3
  ens::RMSProp opt(0.32, 32, 0.88, 1e-8, trainData.n_cols * maxEpochs, -100);
  #else
  // Earlier ensmallen versions did not adjust the step size for the batch size.
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols * maxEpochs, -100);
  #endif
  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1));
  }

  size_t correct = accu(prediction == testLabels);

  double classificationError = 1 - double(correct) / testData.n_cols;
  REQUIRE(classificationError <= classificationErrorThreshold);
}

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::mat, typename ModelType>
void CheckCopyFunction(ModelType* network1,
                       MatType& trainData,
                       MatType& trainLabels)
{
  #if ENS_VERSION_MAJOR >= 3
  ens::RMSProp opt(0.32, 32, 0.88, 1e-8, trainData.n_cols, -1);
  #else
  // Earlier ensmallen versions did not adjust the step size for the batch size.
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols, -1);
  #endif
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);

  FFN<> network2;
  network2 = *network1;
  delete network1;

  // Deallocating all of network1's memory, so that network2 does not use any
  // of that memory.
  arma::mat predictions2;
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::mat, typename ModelType>
void CheckMoveFunction(ModelType* network1,
                       MatType& trainData,
                       MatType& trainLabels)
{
  #if ENS_VERSION_MAJOR >= 3
  ens::RMSProp opt(0.32, 32, 0.88, 1e-8, trainData.n_cols, -1);
  #else
  // Earlier ensmallen versions did not adjust the step size for the batch size.
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols, -1);
  #endif
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);
  FFN<> network2(std::move(*network1));
  delete network1;

  // Deallocating all of network1's memory, so that network2 does not use any
  // of that memory.
  arma::mat predictions2;
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

/**
 * Check whether copying and moving Vanila network is working or not.
 */
TEST_CASE("CheckCopyMovingVanillaNetworkTest", "[FeedForwardNetworkTest][tiny]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  // Normalize labels to [0, 2].
  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
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

  FFN<NegativeLogLikelihood> *model = new FFN<NegativeLogLikelihood>;
  model->Add<Linear>(8);
  model->Add<Sigmoid>();
  model->Add<Linear>(3);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>;
  model1->Add<Linear>(8);
  model1->Add<Sigmoid>();
  model1->Add<Linear>(3);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels);
}

/**
 * Check whether copying and moving network with linear3d is working or not.
 */
TEST_CASE("CheckCopyMovingLinear3DNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, Fatal + Transpose);

  // Normalize labels to [0, 2].
  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
  trainData.shed_row(trainData.n_rows - 1);

  // Construct a feed forward network with trainData.n_rows input nodes,
  // followed by a linear layer and then a Linear3D layer.
  FFN<NegativeLogLikelihood> *model = new FFN<NegativeLogLikelihood>;
  model->Add<Linear>(8);
  model->Add<Sigmoid>();
  model->Add<Linear3D>(3);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>;
  model1->Add<Linear>(8);
  model1->Add<Sigmoid>();
  model1->Add<Linear3D>(3);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels);
}

/**
 * Check whether copying and moving of Noisy Linear layer is working or not.
 */
TEST_CASE("CheckCopyMovingNoisyLinearTest", "[FeedForwardNetworkTest]")
{
  // Create training input by 10x1 matrix (only 1 point).
  arma::mat input = arma::randu(10, 1);
  // Create training output by 1-point matrix.
  arma::mat output = arma::mat("0");

  // Check copying constructor.
  FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>();
  model1->ResetData(input, output);
  model1->Add<NoisyLinear>(5);
  model1->Add<Linear>(1);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model1, input, output);

  // Check moving constructor.
  FFN<NegativeLogLikelihood> *model2 = new FFN<NegativeLogLikelihood>();
  model2->ResetData(input, output);
  model2->Add<NoisyLinear>(5);
  model2->Add<Linear>(1);
  model2->Add<LogSoftMax>();

  // Check whether move constructor is working or not.
  CheckMoveFunction(model2, input, output);
}

/**
 * Check whether copying and moving of concatenate layer is working or not.
 */
TEST_CASE("CheckCopyMovingConcatenateTest", "[FeedForwardNetworkTest]")
{
  // Create training input by 5x5 matrix.
  arma::mat input = arma::randu(10, 1);
  // Create training output by 1 matrix.
  arma::mat output = arma::mat("1");

  // Check copying constructor.
  FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>();
  model1->ResetData(input, output);
  model1->Add<Linear>(5);

  // Create concatenate layer.
  arma::mat concatMatrix = arma::ones(5, 1);
  Concatenate concatLayer;
  concatLayer.Concat() = concatMatrix;

  // Add concatenate layer to the current network.
  model1->Add(std::move(concatLayer));
  model1->Add<Linear>(5);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model1, input, output);

  // Check moving constructor.
  FFN<NegativeLogLikelihood> *model2 = new FFN<NegativeLogLikelihood>();
  model2->ResetData(input, output);
  model2->Add<Linear>(5);

  // Create new concat layer.
  Concatenate concatLayer2;
  concatLayer2.Concat() = concatMatrix;

  // Add concatenate layer to the current network.
  model2->Add(std::move(concatLayer2));
  model2->Add<Linear>(5);
  model2->Add<LogSoftMax>();

  // Check whether move constructor is working or not.
  CheckMoveFunction(model2, input, output);
}

/**
 * Check whether copying and moving of Dropout network is working or not.
 */
TEST_CASE("CheckCopyMovingDropoutNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, Fatal + Transpose);

  // Normalize labels to [0, 2].
  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
  trainData.shed_row(trainData.n_rows - 1);

  FFN<NegativeLogLikelihood> *model = new FFN<NegativeLogLikelihood>;
  model->Add<Linear>(8);
  model->Add<Sigmoid>();
  model->Add<Dropout>(0.3);
  model->Add<Linear>(3);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>;
  model1->Add<Linear>(8);
  model1->Add<Sigmoid>();
  model1->Add<Dropout>(0.3);
  model1->Add<Linear>(3);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels);
}

/**
 * Check whether copying and moving Vanila network is working or not.
 */
TEST_CASE("CheckCopyMovingVanillaNetworkTestNoBias", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  // Normalize labels to [0, 2].
  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
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
   * |     |       |     |       |     |
   * +-----+       +--+--+       +-----+
   */

  FFN<NegativeLogLikelihood> *model = new FFN<NegativeLogLikelihood>;
  model->Add<LinearNoBias>(8);
  model->Add<Sigmoid>();
  model->Add<LinearNoBias>(3);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>;
  model1->Add<LinearNoBias>(8);
  model1->Add<Sigmoid>();
  model1->Add<LinearNoBias>(3);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction<>(model, trainData, trainLabels);

  // Check whether move constructor is working or not.
  CheckMoveFunction<>(model1, trainData, trainLabels);
}

/**
 * Train the vanilla network on a larger dataset.
 */
TEST_CASE("FFVanillaNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);
  trainLabels -= 1; // Labels should be from 0 to numClasses - 1.

  arma::mat testData;
  if (!data::Load("thyroid_test.csv", testData))
    FAIL("Cannot load dataset thyroid_test.csv");

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);
  testLabels -= 1; // Labels should be from 0 to numClasses - 1.

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

  FFN<NegativeLogLikelihood> model;
  model.Add<Linear>(8);
  model.Add<Sigmoid>();
  model.Add<Linear>(3);
  model.Add<LogSoftMax>();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  double error = TestClassificationNetwork(model, trainData, trainLabels,
      testData, testLabels, 10);
  REQUIRE(error <= 0.1);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.csv");
  // Make sure the data loaded okay.
  REQUIRE(!dataset.is_empty());

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  FFN<NegativeLogLikelihood> model1;
  model1.Add<Linear>(10);
  model1.Add<Sigmoid>();
  model1.Add<Linear>(2);
  model1.Add<LogSoftMax>();
  // Vanilla neural net with logistic activation function.
  error = TestClassificationNetwork(model1, dataset, labels, dataset, labels,
      10);
  REQUIRE(error <= 0.2);
}

TEST_CASE("ForwardBackwardTest", "[FeedForwardNetworkTest][long]")
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.csv");
  // Make sure the data loaded okay.
  REQUIRE(!dataset.is_empty());

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  FFN<NegativeLogLikelihood> model;
  model.Add<Linear>(50);
  model.Add<Sigmoid>();
  model.Add<Linear>(10);
  model.Add<LogSoftMax>();

  ens::VanillaUpdate opt;
  #if ENS_VERSION_MAJOR == 1
  opt.Initialize(model.Parameters().n_rows, model.Parameters().n_cols);
  #else
  ens::VanillaUpdate::Policy<arma::mat, arma::mat> optPolicy(opt,
      model.Parameters().n_rows, model.Parameters().n_cols);
  #endif

  #if ENS_VERSION_MAJOR >= 3
  double stepSize = 0.1;
  #else
  // Older versions of ensmallen did not adjust the step size with the batch
  // size.
  double stepSize = 0.01;
  #endif
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
            arma::max(currentResuls.col(i)) == currentResuls.col(i), 1));
      }

      size_t correct = accu(prediction == currentLabels);
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
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);
  trainLabels -= 1; // Labels should be from 0 to numClasses - 1.

  arma::mat testData;
  if (!data::Load("thyroid_test.csv", testData))
    FAIL("Cannot load dataset thyroid_test.csv");

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);
  testLabels -= 1; // Labels should be from 0 to numClasses - 1.

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

  FFN<NegativeLogLikelihood> model;
  model.Add<Linear>(8);
  model.Add<Sigmoid>();
  model.Add<Dropout>();
  model.Add<Linear>(3);
  model.Add<LogSoftMax>();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  double error = TestClassificationNetwork(model, trainData, trainLabels,
      testData, testLabels, 10);
  REQUIRE(error <= 0.1);
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.csv");
  // Make sure the data loaded okay.
  REQUIRE(!dataset.is_empty());

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    dataset.col(i) /= norm(dataset.col(i), 2);
  }

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  FFN<NegativeLogLikelihood> model1;
  model1.Add<Linear>(10);
  model1.Add<Sigmoid>();
  model.Add<Dropout>();
  model1.Add<Linear>(2);
  model1.Add<LogSoftMax>();
  // Vanilla neural net with logistic activation function.
  error = TestClassificationNetwork(model1, dataset, labels, dataset, labels,
      10);
  REQUIRE(error <= 0.2);
}

/**
 * Train the DropConnect network on a larger dataset.
 */
TEST_CASE("DropConnectNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);
  trainLabels -= 1; // The range should be between 0 and numClasses - 1.

  arma::mat testData;
  if (!data::Load("thyroid_test.csv", testData))
    FAIL("Cannot load dataset thyroid_test.csv");

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);
  testLabels -= 1; // The range should be between 0 and numClasses - 1.

 /*
  *  Construct a feed forward network with trainData.n_rows input nodes,
  *  hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
  *  network struct that looks like:
  *
  *  Input         Hidden     DropConnect     Output
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
  *
  *
  */

  FFN<NegativeLogLikelihood> model;
  model.Add<Linear>(8);
  model.Add<Sigmoid>();
  model.Add<DropConnect>(3);
  model.Add<LogSoftMax>();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  double error = TestClassificationNetwork(model, trainData, trainLabels,
      testData, testLabels, 10);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.csv");
  // Make sure the data loaded okay.
  REQUIRE(!dataset.is_empty());

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  FFN<NegativeLogLikelihood> model1;
  model1.Add<Linear>(10);
  model1.Add<Sigmoid>();
  model1.Add<DropConnect>(2);
  model1.Add<LogSoftMax>();

  // Vanilla neural net with logistic activation function.
  error = TestClassificationNetwork(model1, dataset, labels, dataset, labels,
      10);
  REQUIRE(error <= 0.2);
}

/**
 * Test miscellaneous things of FFN,
 * e.g. copy/move constructor, assignment operator.
 */
TEST_CASE("FFNMiscTest", "[FeedForwardNetworkTest]")
{
  FFN<MeanSquaredError> model;
  model.Add<Linear>(3);
  model.Add<ReLU>();

  auto copiedModel(model);
  copiedModel = model;
  auto movedModel(std::move(model));
  auto moveOperator = std::move(copiedModel);
}

/**
 * Test that serialization works ok.
 */
TEST_CASE("FFSerializationTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);
  trainLabels -= 1; // The labels should be between 0 and numClasses - 1.

  arma::mat testData;
  if (!data::Load("thyroid_test.csv", testData))
    FAIL("Cannot load dataset thyroid_test.csv");

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);
  testLabels -= 1; // The labels should be between 0 and numClasses - 1.

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  FFN<NegativeLogLikelihood> model;
  model.Add<Linear>(8);
  model.Add<Sigmoid>();
  model.Add<Dropout>();
  model.Add<Linear>(3);
  model.Add<LogSoftMax>();

  #if ENS_VERSION_MAJOR >= 3
  ens::RMSProp opt(0.32, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);
  #else
  // Older versions of ensmallen did not adjust for the batch size.
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);
  #endif

  model.Train(trainData, trainLabels, opt);

  FFN<NegativeLogLikelihood> xmlModel, jsonModel, binaryModel;
  xmlModel.Add<Linear>(10); // Layer that will get removed.

  // Serialize into other models.
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  arma::mat predictions, xmlPredictions, jsonPredictions, binaryPredictions;
  model.Predict(testData, predictions);
  xmlModel.Predict(testData, xmlPredictions);
  jsonModel.Predict(testData, jsonPredictions);
  binaryModel.Predict(testData, binaryPredictions);

  CheckMatrices(predictions, xmlPredictions, jsonPredictions,
      binaryPredictions);
}

/**
 * Test the overload of Forward function which allows partial forward pass.
 */
TEST_CASE("PartialForwardTest", "[FeedForwardNetworkTest]")
{
  FFN<NegativeLogLikelihood, RandomInitialization> model;
  model.Add<Linear>(10);

  // Add a new Add<> module which adds a (learnable) constant term to the input.
  Add addModule;
  model.Add(std::move(addModule));

  LinearNoBias linearNoBiasModule(10);
  model.Add(std::move(linearNoBiasModule));

  model.Add<Linear>(10);

  // Set up the network for inputs of dimensionality 10.
  model.Reset(10);

  // Set the parameters of the Add<> module to a matrix of ones.
  model.Network()[1]->Parameters() = arma::ones(10, 1);
  // Set the parameters of the LinearNoBias<> module to a matrix of ones.
  model.Network()[2]->Parameters() = arma::ones(10, 10);

  arma::mat input = arma::ones(10, 1);
  arma::mat output;

  // Forward pass only through the Add module.
  model.Forward(input,
                output,
                1 /* Index of the Add module */,
                1 /* Index of the Add module */);

  // As we only forward pass through Add module, input and output should
  // differ by a matrix of ones.
  CheckMatrices(input, output - 1);

  // Forward pass only through the Add module and the LinearNoBias module.
  model.Forward(input,
                output,
                1 /* Index of the Add module */,
                2 /* Index of the LinearNoBias module */);

  // As we only forward pass through Add module followed by the LinearNoBias
  // module, output should be a matrix of 20s.(output = weight * input)
  CheckMatrices(output, arma::ones(10, 1) * 20);
}

/**
 * Test that FFN::Train() returns finite objective value.
 */
TEST_CASE("FFNTrainReturnObjective", "[FeedForwardNetworkTest][tiny]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);
  trainLabels -= 1; // The labels should be between 0 and numClasses.

  arma::mat testData;
  if (!data::Load("thyroid_test.csv", testData))
    FAIL("Cannot load dataset thyroid_test.csv");

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);
  testLabels -= 1; // The labels should be between 0 and numClasses.

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significantly better than 92%.
  FFN<NegativeLogLikelihood> model;
  model.Add<Linear>(8);
  model.Add<Sigmoid>();
  model.Add<Dropout>();
  model.Add<Linear>(3);
  model.Add<LogSoftMax>();

  #if ENS_VERSION_MAJOR >= 3
  ens::RMSProp opt(0.32, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);
  #else
  // Older versions of ensmallen did not adjust the step size for the batch
  // size.
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);
  #endif

  double objVal = model.Train(trainData, trainLabels, opt);

  REQUIRE(std::isfinite(objVal) == true);
}

/**
 * Test that FFN::Model() allows us to access the instantiated network.
 */
TEST_CASE("FFNReturnModel", "[FeedForwardNetworkTest]")
{
  // Create dummy network.
  FFN<NegativeLogLikelihood> model;
  Linear linearA(3);
  model.Add(std::move(linearA));
  Linear linearB(4);
  model.Add(std::move(linearB));

  // Initialize network parameters, with a new input size of 3.
  model.Reset(3);

  // Set all network parameter to one.
  model.Parameters().ones();

  // Zero the second layer parameter.
  model.Network()[1]->Parameters().zeros();

  // Get the layer parameter from layer A and layer B and store them in
  // parameterA and parameterB.
  const arma::mat parameterA = model.Network()[0]->Parameters();
  const arma::mat parameterB = model.Network()[1]->Parameters();

  CheckMatrices(parameterA, arma::ones(3 * 3 + 3, 1));
  CheckMatrices(parameterB, arma::zeros(3 * 4 + 4, 1));

  CheckMatrices(model.Network()[0]->Parameters(), arma::ones(3 * 3 + 3, 1));
  CheckMatrices(model.Network()[1]->Parameters(), arma::zeros(3 * 4 + 4, 1));
}

/**
 * Test to see if the FFN code compiles when the Optimizer
 * doesn't have the MaxIterations() method.
 */
TEST_CASE("OptimizerTest", "[FeedForwardNetworkTest][long]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);
  trainLabels -= 1; // The labels should be between 0 and numClasses.

  arma::mat testData;
  if (!data::Load("thyroid_test.csv", testData))
    FAIL("Cannot load dataset thyroid_test.csv");

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);
  testLabels -= 1; // The labels should be between 0 and numClasses.

  FFN<NegativeLogLikelihood, RandomInitialization> model;
  model.Add<Linear>(8);
  model.Add<Linear>(3);
  model.Add<LogSoftMax>();

  // Since we are only checking compilation here, we set the tolerance to the
  // very large value of 1e5 so it converges quickly.
  ens::DE opt(200, 1000, 0.6, 0.8, 1e5);
  model.Train(trainData, trainLabels, opt);
}

/**
 * Test to see if an exception is thrown when input with
 * wrong shape is provided to a FFN.
 */
TEST_CASE("FFNCheckInputShapeTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  // Normalize labels to [0, 2].
  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, Fatal + Transpose);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  FFN<NegativeLogLikelihood, RandomInitialization> model;
  model.Add<Linear>(8);
  model.Add<Linear>(3);
  model.Add<LogSoftMax>();

  ens::DE opt(200, 1000, 0.6, 0.8, 1e-5);

  // Now set up the input incorrectly.
  model.InputDimensions() = std::vector<size_t>({ 1, 2, 3 });

  REQUIRE_THROWS_AS(model.Train(trainData, trainLabels, opt), std::logic_error);
}

/**
 * Train the RBF network on a larger dataset.
 */
TEST_CASE("RBFNetworkTest", "[FeedForwardNetworkTest][long]")
{
  // Load the dataset.
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat trainLabels1 = arma::zeros(3, trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    trainLabels1.col(i).row(trainLabels(i)) = 1;
  }

  arma::mat testData;
  if (!data::Load("thyroid_test.csv", testData))
    FAIL("Cannot open thyroid_test.csv");

  arma::mat testLabels = testData.row(testData.n_rows - 1) - 1;
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

  FFN<MeanSquaredError> model;
  model.Add<RBF>(8, centroids);
  model.Add<Linear>(3);

  // RBFN neural net with MeanSquaredError.
  double error = TestClassificationNetwork(model, trainData, trainLabels1,
      testData, testLabels, 10);
  REQUIRE(error <= 0.1);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.csv");
  // Make sure the data loaded okay.
  REQUIRE(!dataset.is_empty());

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

  arma::mat centroids1;
  arma::Row<size_t> assignments;
  KMeans<> kmeans1;
  kmeans1.Cluster(dataset, 140, centroids1);

  FFN<MeanSquaredError> model1;
  model1.Add<RBF>(140, centroids1, 4.1);
  model1.Add<Linear>(2);

  // RBFN neural net with MeanSquaredError.
  error = TestClassificationNetwork(model1, dataset, labels1, dataset, labels,
      10);
  REQUIRE(error <= 0.1);
}
