/**
 * @file tests/dag_network_test.cpp
 * @author Andrew Furey
 *
 * Tests the DAGNetwork.
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

#include "../catch.hpp"
#include "../serialization.hpp"

using namespace mlpack;

template<typename MatType = arma::mat, typename ModelType>
void TestNetwork(ModelType& model,
                 MatType& trainData,
                 MatType& trainLabels,
                 MatType& testData,
                 MatType& testLabels,
                 const size_t maxEpochs,
                 const double classificationErrorThreshold)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols * maxEpochs, -100);
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

template<typename MatType = arma::mat, typename ModelType>
void CheckCopy(ModelType* network1,
               MatType& trainData,
               MatType& trainLabels)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols, -1);
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);

  ModelType network2;
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
void CheckMove(ModelType* network1,
               MatType& trainData,
               MatType& trainLabels)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols, -1);
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);
  ModelType network2(std::move(*network1));
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
TEST_CASE("CheckCopyVanillaDAGNetworkTest", "[DAGNetworkTest]")
{
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
  trainData.shed_row(trainData.n_rows - 1);

  DAGNetwork<> *model = new DAGNetwork<>();
  size_t layer1 = model->Add<Linear>(8);
  size_t layer2 = model->Add<Sigmoid>();
  size_t layer3 = model->Add<Linear>(3);
  size_t layer4 = model->Add<LogSoftMax>();
  model->Connect(layer1, layer2);
  model->Connect(layer2, layer3);
  model->Connect(layer3, layer4);

  CheckCopy(model, trainData, trainLabels);
}

TEST_CASE("CheckMoveVanillaDAGNetworkTest", "[DAGNetworkTest]")
{
  arma::mat trainData;
  if (!data::Load("thyroid_train.csv", trainData))
    FAIL("Cannot open thyroid_train.csv");

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
  trainData.shed_row(trainData.n_rows - 1);

  DAGNetwork<> *model = new DAGNetwork();
  size_t layer1 = model->Add<Linear>(8);
  size_t layer2 = model->Add<Sigmoid>();
  size_t layer3 = model->Add<Linear>(3);
  size_t layer4 = model->Add<LogSoftMax>();
  model->Connect(layer1, layer2);
  model->Connect(layer2, layer3);
  model->Connect(layer3, layer4);

  CheckMove(model, trainData, trainLabels);
}

TEST_CASE("DAGNetworkSetAxisOnNonExistentLayer", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  model.Add<Linear>(2); // id == 0

  REQUIRE_THROWS_AS(model.SetAxis(1, 0), std::logic_error); // id == 1
}

TEST_CASE("DAGNetworkConnectNonExistentChild", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  size_t a = model.Add<Linear>(2);
  size_t b = model.Add<Linear>(2);

  model.Connect(a, b);
  REQUIRE_THROWS_AS(model.Connect(b, 2), std::logic_error);
}

template <typename MatType>
using LinearMatType = Linear<MatType>;

template<typename MatType>
using ExampleModel =
  DAGNetwork<MeanSquaredError, RandomInitialization, MatType>;

TEST_CASE("DAGNetworkUseNetworkMatType", "[DAGNetworkTest]")
{
  ExampleModel<arma::mat> model;
  model.InputDimensions() = { 6 };

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  size_t a = model.Add<LinearMatType>(2);
  size_t b = model.Add<LinearMatType>(2);
  model.Connect(a, b);

  REQUIRE_NOTHROW(model.Predict(testInput, testOutput));
}

void CheckConcatenation(arma::mat& input,
                        arma::mat& expectedOutput,
                        std::vector<size_t>& inputDimensions,
                        size_t axis)
{
  DAGNetwork<MeanSquaredError, ConstInitialization> model =
    DAGNetwork(MeanSquaredError(), ConstInitialization(1.0f));

  model.InputDimensions() = inputDimensions;

  arma::mat testOutput;

  size_t a = model.Add<Identity>();
  size_t b = model.Add<Identity>();
  size_t c = model.Add<Identity>();
  model.Connect(a, b);
  model.Connect(b, c);
  model.Connect(a, c);
  model.SetAxis(c, axis);

  model.Predict(input, testOutput);
  CheckMatrices(testOutput, expectedOutput);
}

TEST_CASE("DAGNetworkTestConcatAxis0", "[DAGNetworkTest]")
{
  size_t axis = 0;
  arma::mat input = arma::mat({
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  }).t();

  arma::mat expectedOutput = arma::mat({
    0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7,
    8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15
  }).t();

  std::vector<size_t> inputDimensions = { 2, 2, 2, 2 };
  CheckConcatenation(input, expectedOutput, inputDimensions, axis);
}

TEST_CASE("DAGNetworkTestConcatAxis1", "[DAGNetworkTest]")
{
  size_t axis = 1;
  arma::mat input = arma::mat({
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  }).t();

  arma::mat expectedOutput = arma::mat({
    0, 1, 2, 3, 0, 1, 2, 3,
    4, 5, 6, 7, 4, 5, 6, 7,
    8, 9, 10, 11, 8, 9, 10, 11,
    12, 13, 14, 15, 12, 13, 14, 15
  }).t();

  std::vector<size_t> inputDimensions = { 2, 2, 2, 2 };
  CheckConcatenation(input, expectedOutput, inputDimensions, axis);
}

TEST_CASE("DAGNetworkTestConcatAxis2", "[DAGNetworkTest]")
{
  size_t axis = 2;
  arma::mat input = arma::mat({
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  }).t();

  arma::mat expectedOutput = arma::mat({
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    8, 9, 10, 11, 12, 13, 14, 15
  }).t();

  std::vector<size_t> inputDimensions = { 2, 2, 2, 2 };
  CheckConcatenation(input, expectedOutput, inputDimensions, axis);
}

TEST_CASE("DAGNetworkTestConcatAxis3", "[DAGNetworkTest]")
{
  size_t axis = 3;
  arma::mat input = arma::mat({
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  }).t();

  arma::mat expectedOutput = arma::mat({
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15
  }).t();

  std::vector<size_t> inputDimensions = { 2, 2, 2, 2 };
  CheckConcatenation(input, expectedOutput, inputDimensions, axis);
}

TEST_CASE("DAGNetworkConnectNonExistentParent", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  size_t a = model.Add<Linear>(2);
  size_t b = model.Add<Linear>(2);

  model.Connect(a, b);
  REQUIRE_THROWS_AS(model.Connect(a, 2), std::logic_error);
}

TEST_CASE("DAGNetworkParentIsChild", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  size_t parent = model.Add<Linear>(2);

  REQUIRE_THROWS_AS(model.Connect(parent, parent), std::logic_error);
}

TEST_CASE("DAGNetworkCheckEmptyGraphTest", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput),
                    std::invalid_argument);
}

TEST_CASE("DAGNetworkCheckNoOutputLayers", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);
  model.Connect(layer2, layer3);
  model.Connect(layer3, layer0);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkCheckNoInputLayers", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);
  model.Connect(layer2, layer0);
  model.Connect(layer2, layer3);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkCheckCycle", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);
  size_t layer4 = model.Add<Linear>(10);
  size_t layer5 = model.Add<Linear>(10);

  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);
  model.Connect(layer2, layer3);
  model.Connect(layer3, layer4);
  model.Connect(layer4, layer2);
  model.Connect(layer3, layer5);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkNoAxisForConcatenation", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer0, layer1);
  model.Connect(layer0, layer2);
  model.Connect(layer2, layer3);
  model.Connect(layer1, layer3);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkConcatenationAxisOutOfBounds", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(2);
  size_t layer1 = model.Add<Linear>(2);
  size_t layer2 = model.Add<Linear>(5);
  size_t layer3 = model.Add<Linear>(5);

  model.Connect(layer0, layer1);
  model.Connect(layer0, layer2);
  model.Connect(layer2, layer3);
  model.Connect(layer1, layer3);
  model.SetAxis(layer3, 1);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkComputeOutputDimensions", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 7, 7, 3 };

  size_t layer0 = model.Add<Convolution>(1, 3, 3);
  size_t layer1 = model.Add<Convolution>(3, 3, 3, 1, 1, 1, 1);
  size_t layer2 = model.Add<Convolution>(1, 3, 3, 1, 1, 1, 1);
  size_t layer3 = model.Add<Convolution>(1, 5, 5);

  model.Connect(layer0, layer1);
  model.Connect(layer0, layer2);
  model.Connect(layer2, layer3);
  model.Connect(layer1, layer3);
  model.SetAxis(layer3, 2);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  std::vector<size_t> expectedOutputDimensions = { 1, 1, 1 };
  REQUIRE(model.OutputDimensions() == expectedOutputDimensions);
}

TEST_CASE("DAGNetworkIncorrectInputDimensions", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 5 };

  model.Add<Linear>(4);

  arma::mat testInput = arma::ones(6); // 5 != 6
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkWrongDimensionsForConcat", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 13, 13, 3 };

  size_t layer0 = model.Add<Identity>();
  size_t layer1 = model.Add<Convolution>(1, 3, 3);
  size_t layer2 = model.Add<Convolution>(3, 3, 3);
  size_t layer3 = model.Add<Identity>();

  model.Connect(layer0, layer1);
  model.Connect(layer0, layer2);
  model.Connect(layer2, layer3);
  model.Connect(layer1, layer3);
  model.SetAxis(layer3, 1);

  arma::mat testInput = arma::ones(13 * 13 * 3);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkMultipleOutputsTest", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer1, layer3);
  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkMultipleInputsTest", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer0, layer2);
  model.Connect(layer1, layer2);
  model.Connect(layer2, layer3);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkForestTest", "[DAGNetworkTest]")
{
  DAGNetwork model;
  model.InputDimensions() = { 6 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);

  size_t layer3 = model.Add<Linear>(10);
  size_t layer4 = model.Add<Linear>(10);
  size_t layer5 = model.Add<Linear>(10);

  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);

  model.Connect(layer3, layer4);
  model.Connect(layer4, layer5);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkDiamondTest", "[DAGNetworkTest]")
{
   /*
    *
    *    Example that an FFN could not do.
    *
    *         -- B ----
    *        /     \   \
    *      A ------ C -- D
    * 
    */

  DAGNetwork dagnet;

  dagnet.InputDimensions() = { 6 };

  size_t layerA = dagnet.Add<Linear>(4);
  size_t layerB = dagnet.Add<Linear>(4);
  size_t layerC = dagnet.Add<Linear>(4);
  size_t layerD = dagnet.Add<Linear>(4);

  dagnet.Connect(layerA, layerC);
  dagnet.Connect(layerA, layerB);
  dagnet.Connect(layerB, layerC);
  dagnet.SetAxis(layerC, 0);

  dagnet.Connect(layerB, layerD);
  dagnet.Connect(layerC, layerD);
  dagnet.SetAxis(layerD, 0);


  arma::mat testInput = arma::ones(6);
  arma::mat dagnetOutput;

  REQUIRE_NOTHROW(dagnet.Predict(testInput, dagnetOutput));
  std::vector<size_t> expectedOutputDims = { 4 };
  REQUIRE(dagnet.OutputDimensions() == expectedOutputDims);
}

template <typename MatType>
using Model = DAGNetwork<MeanSquaredError, RandomInitialization, MatType>;

template<template<typename> typename ModelType = Model,
        typename MatType = arma::mat>
size_t AddLinearRelu(ModelType<MatType>& model) {
  MultiLayer<MatType> hiddenLayer;
  hiddenLayer.template Add<Linear>(2);
  hiddenLayer.template Add<LeakyReLU>(0.1f);

  return model.Add(hiddenLayer);
}

TEST_CASE("DAGNetworkAddMultiLayer", "[DAGNetworkTest]")
{
  Model<arma::mat> model;

  size_t a = AddLinearRelu<Model, arma::mat>(model);
  size_t b = AddLinearRelu<Model, arma::mat>(model);
  size_t c = AddLinearRelu<Model, arma::mat>(model);

  model.Connect(a, b);
  model.Connect(b, c);

  arma::mat actual = arma::mat({ { 1.5f, -2.0f} }).t();
  arma::mat input  = arma::mat({ { 4.5f, -2.0f} }).t();

  REQUIRE_NOTHROW(model.Evaluate(input, actual));
}

TEST_CASE("DAGNetworkGradientAccumulatesAndResetsToZero", "[DAGNetworkTest]")
{
  DAGNetwork<MeanSquaredError, ConstInitialization> model =
    DAGNetwork(MeanSquaredError(), ConstInitialization(1.0f));

  size_t a = model.Add<Add>();
  size_t b = model.Add<Add>();
  size_t c = model.Add<Add>();

  model.Connect(a, b);
  model.Connect(a, c);
  model.Connect(b, c);
  model.SetAxis(c, 0);

  arma::mat input  = arma::ones(2);
  arma::mat actual = arma::ones(4);
  arma::mat output;

  arma::mat deltaLoss = arma::ones(4);
  arma::mat gradients;
  arma::mat expectedGradients = arma::mat({2, 2, 1, 1, 1, 1, 1, 1}).t();
  MeanSquaredError lossFunction;

  model.Forward(input, output);
  model.Backward(input, output, deltaLoss, gradients);

  CheckMatrices(gradients, expectedGradients);

  // Gradients at layer a should have been set to zero.
  model.Forward(input, output);
  model.Backward(input, output, deltaLoss, gradients);
  CheckMatrices(gradients, expectedGradients);
}

TEST_CASE("DAGNetworkSerializationTest", "[DAGNetworkTest]")
{
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
  DAGNetwork model;
  size_t linear_layer_1 = model.Add<Linear>(8);
  size_t sigmoid_layer_2 = model.Add<Sigmoid>();
  size_t dropout_layer_3 = model.Add<Dropout>();
  size_t linear_layer_4 = model.Add<Linear>(3);
  size_t logsoftmax_layer_5 = model.Add<LogSoftMax>();

  model.Connect(linear_layer_1, sigmoid_layer_2);
  model.Connect(sigmoid_layer_2, dropout_layer_3);
  model.Connect(dropout_layer_3, linear_layer_4);
  model.Connect(linear_layer_4, logsoftmax_layer_5);

  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);

  model.Train(trainData, trainLabels, opt);

  DAGNetwork xmlModel, jsonModel, binaryModel;
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
