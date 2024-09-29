/**
 * @file tests/recurrent_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the recurrent network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

#include <ensmallen.hpp>

#include "../catch.hpp"
#include "../serialization.hpp"

using namespace mlpack;
using namespace ens;

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
/* TEST_CASE("LSTMDistractedSequenceRecallTest", "[RecurrentNetworkTest]") */
/* { */
/*   DistractedSequenceRecallTestNetwork<LSTM<> >(4, 8); */
/* } */

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
/* TEST_CASE("FastLSTMDistractedSequenceRecallTest", "[RecurrentNetworkTest]") */
/* { */
/*   DistractedSequenceRecallTestNetwork<FastLSTM<> >(4, 8); */
/* } */

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
/* TEST_CASE("GRUDistractedSequenceRecallTest", "[RecurrentNetworkTest]") */
/* { */
/*   DistractedSequenceRecallTestNetwork<GRU<> >(4, 8); */
/* } */

/**
 * Ensure fast LSTMs work with larger batch sizes.
 */
//TEST_CASE("FastLSTMBatchSizeTest", "[RecurrentNetworkTest]")
//{
//  BatchSizeTest<FastLSTM<>>();
//}

/**
 * Ensure GRUs work with larger batch sizes.
 */
//TEST_CASE("GRUBatchSizeTest", "[RecurrentNetworkTest]")
//{
//  BatchSizeTest<GRU<>>();
//}

/**
 * Make sure the RNN can be properly serialized.
 *
TEST_CASE("RNNSerializationTest", "[RecurrentNetworkTest]")
{
  const size_t rho = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::cube input;
  arma::mat labelsTemp;
  GenerateNoisySines(input, labelsTemp, rho, 6);

  arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
    labels.tube(0, i).fill(value);
  }

  /**
   * Construct a network with 1 input unit, 4 hidden units and 10 output
   * units. The hidden layer is connected to itself. The network structure
   * looks like:
   *
   *  Input         Hidden        Output
   * Layer(1)      Layer(4)      Layer(10)
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   *
  Add<> add(4);
  Linear<> lookup(1, 4);
  SigmoidLayer<> sigmoidLayer;
  Linear<> linear(4, 4);
  Recurrent<>* recurrent = new Recurrent<>(add, lookup, linear,
      sigmoidLayer, rho);

  RNN<> model(rho);
  model.Add<IdentityLayer<> >();
  model.Add(recurrent);
  model.Add<Linear<> >(4, 10);
  model.Add<LogSoftMax<> >();

  StandardSGD opt(0.1, 1, input.n_cols /* 1 epoch *, -100);
  model.Train(input, labels, opt);

  // Serialize the network.
  RNN<> xmlModel(1), jsonModel(3), binaryModel(5);
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  // Take predictions, check the output.
  arma::cube prediction, xmlPrediction, jsonPrediction, binaryPrediction;
  model.Predict(input, prediction);
  xmlModel.Predict(input, xmlPrediction);
  jsonModel.Predict(input, jsonPrediction);
  binaryModel.Predict(input, binaryPrediction);

  CheckMatrices(prediction, xmlPrediction, jsonPrediction, binaryPrediction);
}
*/

/**
 * Train the BRNN on a larger dataset.
 *
TEST_CASE("SequenceClassificationBRNNTest", "[RecurrentNetworkTest]")
{
  // Using same test for RNN below.
  size_t successes = 0;
  const size_t rho = 10;

  for (size_t trial = 0; trial < 6; ++trial)
  {
    // Generate 12 (2 * 6) noisy sines. A single sine contains rho
    // points/features.
    arma::cube input;
    arma::mat labelsTemp;
    GenerateNoisySines(input, labelsTemp, rho, 6);

    arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
    for (size_t i = 0; i < labelsTemp.n_cols; ++i)
    {
      const int value = arma::as_scalar(arma::find(
          arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1));
      labels.tube(0, i).fill(value);
    }

    Add<> add(4);
    Linear<> lookup(1, 4);
    SigmoidLayer<> sigmoidLayer;
    Linear<> linear(4, 4);
    Recurrent<>* recurrent = new Recurrent<>(
        add, lookup, linear, sigmoidLayer, rho);

    BRNN<> model(rho);
    model.Add<IdentityLayer<> >();
    model.Add(recurrent);
    model.Add<Linear<> >(4, 5);

    StandardSGD opt(0.1, 1, 500 * input.n_cols, -100);
    model.Train(input, labels, opt);
    INFO("Training over");
    arma::cube prediction;
    model.Predict(input, prediction);
    INFO("Prediction over");

    size_t error = 0;
    for (size_t i = 0; i < prediction.n_cols; ++i)
    {
      const int predictionValue = arma::as_scalar(arma::find(
          arma::max(prediction.slice(rho - 1).col(i)) ==
          prediction.slice(rho - 1).col(i), 1));

      const int targetValue = arma::as_scalar(arma::find(
          arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1));

      if (predictionValue == targetValue)
      {
        error++;
      }
    }

    double classificationError = 1 - double(error) / prediction.n_cols;
    INFO(classificationError);
    if (classificationError <= 0.2)
    {
      ++successes;
      break;
    }
  }

  REQUIRE(successes >= 1);
}
*/

/**
 * Train the vanilla network on a larger dataset.
 *
TEST_CASE("SequenceClassificationTest", "[RecurrentNetworkTest]")
{
  // It isn't guaranteed that the recurrent network will converge in the
  // specified number of iterations using random weights. If this works 1 of 6
  // times, I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  size_t successes = 0;
  const size_t rho = 10;

  for (size_t trial = 0; trial < 6; ++trial)
  {
    // Generate 12 (2 * 6) noisy sines. A single sine contains rho
    // points/features.
    arma::cube input;
    arma::mat labelsTemp;
    GenerateNoisySines(input, labelsTemp, rho, 6);

    arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
    for (size_t i = 0; i < labelsTemp.n_cols; ++i)
    {
      const int value = arma::as_scalar(arma::find(
          arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1));
      labels.tube(0, i).fill(value);
    }

    /**
     * Construct a network with 1 input unit, 4 hidden units and 10 output
     * units. The hidden layer is connected to itself. The network structure
     * looks like:
     *
     *  Input         Hidden        Output
     * Layer(1)      Layer(4)      Layer(10)
     * +-----+       +-----+       +-----+
     * |     |       |     |       |     |
     * |     +------>|     +------>|     |
     * |     |    ..>|     |       |     |
     * +-----+    .  +--+--+       +-----+
     *            .     .
     *            .     .
     *            .......
     *
    Add<> add(4);
    Linear<> lookup(1, 4);
    SigmoidLayer<> sigmoidLayer;
    Linear<> linear(4, 4);
    Recurrent<>* recurrent = new Recurrent<>(
        add, lookup, linear, sigmoidLayer, rho);

    RNN<> model(rho);
    model.Add<IdentityLayer<> >();
    model.Add(recurrent);
    model.Add<Linear<> >(4, 10);
    model.Add<LogSoftMax<> >();

    StandardSGD opt(0.1, 1, 500 * input.n_cols, -100);
    model.Train(input, labels, opt);

    arma::cube prediction;
    model.Predict(input, prediction);

    size_t error = 0;
    for (size_t i = 0; i < prediction.n_cols; ++i)
    {
      const int predictionValue = arma::as_scalar(arma::find(
          arma::max(prediction.slice(rho - 1).col(i)) ==
          prediction.slice(rho - 1).col(i), 1));

      const int targetValue = arma::as_scalar(arma::find(
          arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1));

      if (predictionValue == targetValue)
      {
        error++;
      }
    }

    double classificationError = 1 - double(error) / prediction.n_cols;
    if (classificationError <= 0.2)
    {
      ++successes;
      break;
    }
  }

  REQUIRE(successes >= 1);
}
*/

/**
 * Test that RNN::Train() returns finite objective value.
 *
TEST_CASE("RNNTrainReturnObjective", "[RecurrentNetworkTest]")
{
  const size_t rho = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::cube input;
  arma::mat labelsTemp;
  GenerateNoisySines(input, labelsTemp, rho, 6);

  arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1));
    labels.tube(0, i).fill(value);
  }

  /**
   * Construct a network with 1 input unit, 4 hidden units and 10 output
   * units. The hidden layer is connected to itself. The network structure
   * looks like:
   *
   *  Input         Hidden        Output
   * Layer(1)      Layer(4)      Layer(10)
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   *
  Add<> add(4);
  Linear<> lookup(1, 4);
  SigmoidLayer<> sigmoidLayer;
  Linear<> linear(4, 4);
  Recurrent<>* recurrent = new Recurrent<>(add, lookup, linear,
      sigmoidLayer, rho);

  RNN<> model(rho);
  model.Add<IdentityLayer<> >();
  model.Add(recurrent);
  model.Add<Linear<> >(4, 10);
  model.Add<LogSoftMax<> >();

  StandardSGD opt(0.1, 1, input.n_cols /* 1 epoch *, -100);
  double objVal = model.Train(input, labels, opt);

  REQUIRE(std::isfinite(objVal) == true);
}
*/

/**
 * Test that BRNN::Train() returns finite objective value.
 *
TEST_CASE("BRNNTrainReturnObjective", "[RecurrentNetworkTest]")
{
  const size_t rho = 10;

  arma::cube input;
  arma::mat labelsTemp;
  GenerateNoisySines(input, labelsTemp, rho, 6);

  arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1));
    labels.tube(0, i).fill(value);
  }

  Add<> add(4);
  Linear<> lookup(1, 4);
  SigmoidLayer<> sigmoidLayer;
  Linear<> linear(4, 4);
  Recurrent<>* recurrent = new Recurrent<>(
      add, lookup, linear, sigmoidLayer, rho);

  BRNN<> model(rho);
  model.Add<IdentityLayer<> >();
  model.Add(recurrent);
  model.Add<Linear<> >(4, 5);

  StandardSGD opt(0.1, 1, 500 * input.n_cols, -100);
  double objVal = model.Train(input, labels, opt);
  INFO("Training over");

  // Test that BRNN::Train() returns finite objective value.
  REQUIRE(std::isfinite(objVal) == true);
}
*/

/**
 * Test to make sure that an error is thrown when input with
 * wrong input shape is provided to a RNN.
 *
TEST_CASE("RNNCheckInputShapeTest", "[RecurrentNetworkTest]")
{
  const size_t rho = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::cube input;
  arma::mat labelsTemp;
  GenerateNoisySines(input, labelsTemp, rho, 6);

  arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
    labels.tube(0, i).fill(value);
  }

  /**
   * Construct a network with 1 input unit, 4 hidden units and 10 output
   * units. The hidden layer is connected to itself. The network structure
   * looks like:
   *
   *  Input         Hidden        Output
   * Layer(1)      Layer(4)      Layer(10)
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   *
  Add<> add(4);
  // Purposely providing wrong input shape of 3.
  // The correct input shape is 1.
  Linear<> lookup(3, 4);
  SigmoidLayer<> sigmoidLayer;
  Linear<> linear(4, 4);
  Recurrent<>* recurrent = new Recurrent<>(add, lookup, linear,
      sigmoidLayer, rho);

  RNN<> model(rho);
  model.Add<IdentityLayer<> >();
  model.Add(recurrent);
  model.Add<Linear<> >(4, 10);
  model.Add<LogSoftMax<> >();

  std::string expectedMsg = "RNN<>::Train(): ";
  expectedMsg += "the first layer of the network expects ";
  expectedMsg += std::to_string(3) + " elements, ";
  expectedMsg += "but the input has " + std::to_string(1) + " dimensions! ";

  StandardSGD opt(0.1, 1, input.n_cols /* 1 epoch *, -100);

  REQUIRE_THROWS_AS(model.Train(input, labels, opt), std::logic_error);
}
*/
