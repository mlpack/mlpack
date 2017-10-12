/**
 * @file feedforward_network_test.cpp
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

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(FeedForwardNetworkTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<typename MatType = arma::mat>
void BuildVanillaNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
                         const size_t outputSize,
                         const size_t hiddenLayerSize,
                         const size_t maxEpochs,
                         const double classificationErrorThreshold)
{
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
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  // RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

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
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildVanillaNetwork<>
      (trainData, trainLabels, testData, testLabels, 3, 8, 10, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  // Vanilla neural net with logistic activation function.
  BuildVanillaNetwork<>
      (dataset, labels, dataset, labels, 2, 10, 10, 0.2);
}

BOOST_AUTO_TEST_CASE(ForwardBackwardTest)
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

  VanillaUpdate opt;
  model.ResetParameters();
  opt.Initialize(model.Parameters().n_rows, model.Parameters().n_cols);
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
      model.Backward(currentLabels, gradients);
      opt.Update(model.Parameters(), stepSize, gradients);
      batchStart = batchEnd;

      arma::mat prediction = arma::zeros<arma::mat>(1, currentResuls.n_cols);

      for (size_t i = 0; i < currentResuls.n_cols; ++i)
      {
        prediction(i) = arma::as_scalar(arma::find(
            arma::max(currentResuls.col(i)) == currentResuls.col(i), 1)) + 1;
      }

      size_t correct = 0;
      for (size_t i = 0; i < currentLabels.n_cols; i++)
      {
        if (int(arma::as_scalar(prediction.col(i))) ==
            int(arma::as_scalar(currentLabels.col(i))))
        {
          correct++;
        }
      }

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

  BOOST_REQUIRE(converged);
}

/**
 * Train and evaluate a Dropout network with the specified structure.
 */
template<typename MatType = arma::mat>
void BuildDropoutNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
                         const size_t outputSize,
                         const size_t hiddenLayerSize,
                         const size_t maxEpochs,
                         const double classificationErrorThreshold)
{
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
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);

  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

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
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train the dropout network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(DropoutNetworkTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildDropoutNetwork<>
      (trainData, trainLabels, testData, testLabels, 3, 8, 10, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  // Vanilla neural net with logistic activation function.
  BuildDropoutNetwork<>
      (dataset, labels, dataset, labels, 2, 10, 10, 0.2);
}

/**
 * Train and evaluate a DropConnect network(with a baselayer) with the
 * specified structure.
 */
template<typename MatType = arma::mat>
void BuildDropConnectNetwork(MatType& trainData,
                             MatType& trainLabels,
                             MatType& testData,
                             MatType& testLabels,
                             const size_t outputSize,
                             const size_t hiddenLayerSize,
                             const size_t maxEpochs,
                             const double classificationErrorThreshold)
{
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

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<DropConnect<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);

  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

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
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train the dropconnect network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(DropConnectNetworkTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildDropConnectNetwork<>
      (trainData, trainLabels, testData, testLabels, 3, 8, 10, 0.1);

  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  // Vanilla neural net with logistic activation function.
  BuildDropConnectNetwork<>
      (dataset, labels, dataset, labels, 2, 10, 10, 0.2);
}

/**
 * Test miscellaneous things of FFN,
 * e.g. copy/move constructor, assignment operator.
 */
BOOST_AUTO_TEST_CASE(FFNMiscTest)
{
  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(2, 3);
  model.Add<ReLULayer<>>();

  auto copiedModel(model);
  copiedModel = model;
  auto movedModel(std::move(model));
  movedModel = std::move(copiedModel);
}

/**
 * Test that serialization works ok.
 */
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);

  model.Train(trainData, trainLabels, opt);

  FFN<NegativeLogLikelihood<>> xmlModel, textModel, binaryModel;
  xmlModel.Add<Linear<>>(10, 10); // Layer that will get removed.

  // Serialize into other models.
  SerializeObjectAll(model, xmlModel, textModel, binaryModel);

  arma::mat predictions, xmlPredictions, textPredictions, binaryPredictions;
  model.Predict(testData, predictions);
  xmlModel.Predict(testData, xmlPredictions);
  textModel.Predict(testData, textPredictions);
  textModel.Predict(testData, binaryPredictions);

  CheckMatrices(predictions, xmlPredictions, textPredictions,
      binaryPredictions);
}

BOOST_AUTO_TEST_SUITE_END();
