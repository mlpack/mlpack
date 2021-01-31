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

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <ensmallen.hpp>

#include "catch.hpp"
#include "serialization.hpp"
//#include "custom_layer.hpp"

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
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, 10001, -100);
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
  /* REQUIRE(classificationError <= classificationErrorThreshold); */
}

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::mat, typename ModelType>
void CheckCopyFunction(ModelType* network1,
                       MatType& trainData,
                       MatType& trainLabels,
                       const size_t maxEpochs)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols, -1);
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
                       MatType& trainLabels,
                       const size_t maxEpochs)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols, -1);
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
TEST_CASE("CheckCopyMovingVanillaNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  // Normalize labels to [0, 2].
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
  model->Add<Linear>(trainData.n_rows, 8);
  model->Add<Sigmoid>();
  model->Add<Linear>(8, 3);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood<> > *model1 = new FFN<NegativeLogLikelihood<> >;
  model1->Add<Linear>(trainData.n_rows, 8);
  model1->Add<Sigmoid>();
  model1->Add<Linear>(8, 3);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels, 1);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels, 1);
}

/**
 * Check whether copying and moving network with Reparametrization is working or not.
 */
TEST_CASE("CheckCopyMovingReparametrizationNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  // Construct a feed forward network with trainData.n_rows input nodes,
  // followed by a linear layer and then a reparametrization layer.
  FFN<NegativeLogLikelihood<> > *model = new FFN<NegativeLogLikelihood<> >;
  model->Add<Linear>(trainData.n_rows, 8);
  model->Add<Reparametrization>(4, false, true, 1);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood<> > *model1 = new FFN<NegativeLogLikelihood<> >;
  model1->Add<Linear>(trainData.n_rows, 8);
  model1->Add<Reparametrization>(4, false, true, 1);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels, 1);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels, 1);
}

/**
 * Check whether copying and moving network with linear3d is working or not.
 */
TEST_CASE("CheckCopyMovingLinear3DNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  // Construct a feed forward network with trainData.n_rows input nodes,
  // followed by a linear layer and then a Linear3D layer.
  FFN<NegativeLogLikelihood<> > *model = new FFN<NegativeLogLikelihood<> >;
  model->Add<Linear>(trainData.n_rows, 8);
  model->Add<Sigmoid>();
  model->Add<Linear3D>(8, 3);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood<> > *model1 = new FFN<NegativeLogLikelihood<> >;
  model1->Add<Linear>(trainData.n_rows, 8);
  model1->Add<Sigmoid>();
  model1->Add<Linear3D>(8, 3);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels, 1);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels, 1);
}

/**
 * Check whether copying and moving of Noisy Linear layer is working or not.
 */
TEST_CASE("CheckCopyMovingNoisyLinearTest", "[FeedForwardNetworkTest]")
{
  // Create training input by 5x5 matrix.
  arma::mat input = arma::randu(10,1);
  // Create training output by 1 matrix.
  arma::mat output = arma::mat("1");

  // Check copying constructor.
  FFN<NegativeLogLikelihood<>> *model1 = new FFN<NegativeLogLikelihood<>>();
  model1->Predictors() = input;
  model1->Responses() = output;
  model1->Add<Identity>();
  model1->Add<NoisyLinear>(10, 5);
  model1->Add<Linear>(5, 1);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model1, input, output, 1);

  // Check moving constructor.
  FFN<NegativeLogLikelihood<>> *model2 = new FFN<NegativeLogLikelihood<>>();
  model2->Predictors() = input;
  model2->Responses() = output;
  model2->Add<Identity>();
  model2->Add<NoisyLinear>(10, 5);
  model2->Add<Linear>(5, 1);
  model2->Add<LogSoftMax>();

  // Check whether move constructor is working or not.
  CheckMoveFunction(model2, input, output, 1);
}

/**
 * Check whether copying and moving of concatenate layer is working or not.
 */
TEST_CASE("CheckCopyMovingConcatenateTest", "[FeedForwardNetworkTest]")
{
  // Create training input by 5x5 matrix.
  arma::mat input = arma::randu(10,1);
  // Create training output by 1 matrix.
  arma::mat output = arma::mat("1");

  // Check copying constructor.
  FFN<NegativeLogLikelihood<>> *model1 = new FFN<NegativeLogLikelihood<>>();
  model1->Predictors() = input;
  model1->Responses() = output;
  model1->Add<Identity>();
  model1->Add<Linear>(10, 5);

  // Create concatenate layer.
  arma::mat concatMatrix = arma::ones(5, 1);
  Concatenate* concatLayer = new Concatenate();
  concatLayer->Concat() = concatMatrix;

  // Add concatenate layer to the current network.
  model1->Add(concatLayer);
  model1->Add<Linear>(10, 5);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model1, input, output, 1);

  // Check moving constructor.
  FFN<NegativeLogLikelihood<>> *model2 = new FFN<NegativeLogLikelihood<>>();
  model2->Predictors() = input;
  model2->Responses() = output;
  model2->Add<Identity>();
  model2->Add<Linear>(10, 5);

  // Create new concat layer.
  Concatenate* concatLayer2 = new Concatenate();
  concatLayer2->Concat() = concatMatrix;

  // Add concatenate layer to the current network.
  model2->Add(concatLayer2);
  model2->Add<Linear>(10, 5);
  model2->Add<LogSoftMax>();

  // Check whether move constructor is working or not.
  CheckMoveFunction(model2, input, output, 1);
}

/**
 * Check whether copying and moving of Dropout network is working or not.
 */
TEST_CASE("CheckCopyMovingDropoutNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  FFN<NegativeLogLikelihood<> > *model = new FFN<NegativeLogLikelihood<> >;
  model->Add<Linear>(trainData.n_rows, 8);
  model->Add<Sigmoid>();
  model->Add<Dropout>(0.3);
  model->Add<Linear>(8, 3);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood<> > *model1 = new FFN<NegativeLogLikelihood<> >;
  model1->Add<Linear>(trainData.n_rows, 8);
  model1->Add<Sigmoid>();
  model1->Add<Dropout>(0.3);
  model1->Add<Linear>(8, 3);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels, 1);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels, 1);
}
