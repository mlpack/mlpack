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
