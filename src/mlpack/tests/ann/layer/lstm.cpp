/**
 * @file lstm.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 * @author Ryan Curtin
 *
 * Tests the ann layer modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

// Prepare an LSTM layer.  All weights will be set to zero.
template<typename MatType = arma::mat>
LSTMType<MatType> SetupLSTM(const size_t inputSize,
                            const size_t batchSize,
                            const size_t outputSize,
                            const size_t timeSteps,
                            arma::mat& weights)
{
  LSTM l(outputSize);
  l.InputDimensions() = std::vector<size_t>{ inputSize };
  l.ComputeOutputDimensions();
  l.CurrentStep(0);

  // Set all of the weights to 0.
  weights.zeros(l.WeightSize(), 1);
  l.SetWeights(weights);
  l.ClearRecurrentState(timeSteps, batchSize);

  return l;
}

/**
 * LSTM layer numerical gradient test.
 */
TEST_CASE("GradientLSTMLayerTest", "[ANNLayerTest]")
{
  // LSTM function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(4, 1, 1)),
        target(arma::ones(1, 1, 1))
    {
      const size_t rho = 5;

      model = RNN<MeanSquaredError, RandomInitialization>(rho);
      model.ResetData(input, target);
      model.Add<LSTM>(1);
      model.Add<Linear>(1);
      model.InputDimensions() = std::vector<size_t>{ 4 };
    }

    double Gradient(arma::mat& gradient)
    {
      gradient.zeros(model.Parameters().n_elem, 1);
      return model.EvaluateWithGradient(model.Parameters(), 0, gradient, 1);
    }

    arma::mat& Parameters() { return model.Parameters(); }

    RNN<MeanSquaredError, RandomInitialization> model;
    arma::cube input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::cube, typename ModelType>
void CheckRNNCopyFunction(ModelType* network1,
                          MatType& trainData,
                          MatType& trainLabels,
                          const size_t maxEpochs)
{
  arma::cube predictions1;
  arma::cube predictions2;
  ens::StandardSGD opt(0.1, 1, maxEpochs * trainData.n_slices, -100, false);

  network1->Train(trainData, trainLabels, opt);
  network1->Predict(trainData, predictions1);

  RNN<> network2 = *network1;
  delete network1;

  // Deallocating all of network1's memory, so that network2 does not use any
  // of that memory.
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::cube, typename ModelType>
void CheckRNNMoveFunction(ModelType* network1,
                          MatType& trainData,
                          MatType& trainLabels,
                          const size_t maxEpochs)
{
  arma::cube predictions1;
  arma::cube predictions2;
  ens::StandardSGD opt(0.1, 1, maxEpochs * trainData.n_slices, -100, false);

  network1->Train(trainData, trainLabels, opt);
  network1->Predict(trainData, predictions1);

  RNN<> network2(std::move(*network1));
  delete network1;

  // Deallocating all of network1's memory, so that network2 does not use any
  // of that memory.
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

/**
 * Check whether copying and moving network with LSTM is working or not.
 */
TEST_CASE("CheckCopyMoveLSTMTest", "[ANNLayerTest]")
{
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::ones(1, 1, 5);
  const size_t rho = 5;

  RNN<NegativeLogLikelihood>* model1 = new RNN<NegativeLogLikelihood>(rho);
  model1->ResetData(input, target);
  model1->Add<Linear>(10);
  model1->Add<LSTM>(3);
  model1->Add<LogSoftMax>();

  RNN<NegativeLogLikelihood>* model2 = new RNN<NegativeLogLikelihood>(rho);
  model2->ResetData(input, target);
  model2->Add<Linear>(10);
  model2->Add<LSTM>(3);
  model2->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckRNNCopyFunction<>(model1, input, target, 1);

  // Check whether move constructor is working or not.
  CheckRNNMoveFunction<>(model2, input, target, 1);
}

TEST_CASE("LSTMForwardInputOnlyTest", "[ANNLayerTest]")
{
  // Test that we get the expected results when we weight all recurrent
  // connections to 0, and only use the input gate connection.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);

  // Now re-enable only the input gate input weights (not the recurrent ones).
  l.InputGateWeight().randu();
  // We also set the block input bias; otherwise, the input would be fully
  // zeroed out.
  l.BlockInputBias().ones();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // When using only the input connection, the input is multiplied with the
  // input weights, elementwise multiplied with the hyptan values of the block
  // input bias, and then passed through one more hyptan.
  //
  // The multiplication by 0.5 is because of the elementwise combination of the
  // cell with the output gate.
  arma::mat expectedOutput = 0.5 * tanh(
      tanh(repmat(l.BlockInputBias(), 1, batchSize)) %
      (1.0 / (1.0 + exp(-l.InputGateWeight() * input))));

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardBlockInputOnlyTest", "[ANNLayerTest]")
{
  // Test that we get the expected results when we weight all recurrent
  // connections to 0, and only use the block input gate connection.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 1, weights);

  // Set only the block input weights to have values.
  l.BlockInputWeight().randu();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // When using only the block input connection, the input is multiplied with
  // the block input weights, passed through a tanh nonlinearity, and then
  // halved (because of the elementwise division with the input gate), then
  // passed through another tanh and multipied by 0.5 again due to the output
  // gate.
  arma::mat expectedOutput = 0.5 * tanh(
      0.5 * tanh(l.BlockInputWeight() * input));

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardForgetInputOnlyTest", "[ANNLayerTest]")
{
  // Test that we get the expected result when we weight all recurrent
  // connections to 0, and only allow input to the cell through the forget gate.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);
  l.CurrentStep(1);

  // Set all of the weights to 0.
  l.Parameters().zeros();

  // Set only the block input weights to have values.
  l.ForgetGateWeight().randu();

  // Set the cell to contain all ones.
  l.RecurrentState(l.PreviousStep()).zeros();
  l.RecurrentState(l.CurrentStep()).zeros();
  arma::mat cell;
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  cell.ones();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // When using only the forget gate, the input is multiplied with the forget
  // gate weights, passes through a sigmoid, then a tanh, then is multiplied by
  // 0.5 due to the output gate.
  arma::mat expectedOutput = 0.5 * tanh(
      1.0 / (1.0 + exp(-l.ForgetGateWeight() * input)));

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardIgnoreInputTest", "[ANNLayerTest]")
{
  // Test that we get the expected results when we weight the LSTM to ignore all
  // input and only output the cell.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);
  l.CurrentStep(1);

  // Set the cell state to a random vector.
  arma::mat cell, storedCell;
  storedCell.randu(outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  cell = storedCell;

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // The old cell state should be multiplied by 0.5, then passed through a tanh
  // and multiplied by 0.5 again.
  arma::mat expectedOutput = 0.5 * tanh(0.5 * storedCell);

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardInputPeepholeOnlyTest", "[ANNLayerTest]")
{
  // Set only the input peephole weights to nonzero values.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);
  l.CurrentStep(1);
  l.RecurrentState(l.PreviousStep()).zeros();
  l.RecurrentState(l.CurrentStep()).zeros();

  // The only nonzero parameters are the peephole connection weights to the
  // input gate.
  l.PeepholeInputGateWeight().randu();
  // But we also need to set the bias of the block input to something, otherwise
  // the input connection will end up being zero.
  l.BlockInputBias().ones();

  // For the peephole connection to give any output, we must also set the cell
  // value to something.
  arma::mat cell, storedCell;
  storedCell.randu(outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  cell = storedCell;

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // The peephole connection should connect the cell state back to the input
  // gate, where it will pass through a sigmoid and be multiplied elementwise
  // with tanh(1) (the block input bias).  Then it will have half the cell state
  // added to it, and pass through another tanh and be multiplied by 0.5 (the
  // sigmoid-ed zero-value output gate).
  arma::mat expectedOutput = 0.5 * tanh(
      tanh(repmat(l.BlockInputBias(), 1, batchSize)) %
      (1.0 / (1.0 + exp(
          -repmat(l.PeepholeInputGateWeight(), 1, batchSize) % storedCell))) +
      0.5 * storedCell);

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardForgetPeepholeOnlyTest", "[ANNLayerTest]")
{
  // Set only the forget peephole weights to nonzero values.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);
  l.CurrentStep(1);
  l.RecurrentState(l.PreviousStep()).zeros();
  l.RecurrentState(l.CurrentStep()).zeros();

  // The only nonzero parameters are the peephole connection weights to the
  // forget gate.
  l.PeepholeForgetGateWeight().randu();

  // For the peephole connection to give any output, we must also set the cell
  // value to something.
  arma::mat cell, storedCell;
  storedCell.randu(outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  cell = storedCell;

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // The peephole connection should connect the cell state back to the forget
  // gate, giving a forget gate value of (peepholeWeights % storedCell).  This
  // is then passed through a sigmoid, multiplied elementwise with the cell
  // value, then passed through a hyptan and multiplied by 0.5 (via the output
  // gate value).
  arma::mat expectedOutput = 0.5 * tanh(
      storedCell % (1.0 / (1.0 + exp(
          -repmat(l.PeepholeForgetGateWeight(), 1, batchSize) % storedCell))));

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardOutputPeepholeOnlyTest", "[ANNLayerTest]")
{
  // Set only the output peephole weights to nonzero values.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);
  l.CurrentStep(1);
  l.RecurrentState(l.PreviousStep()).zeros();
  l.RecurrentState(l.CurrentStep()).zeros();

  // The only nonzero parameters are the peephole connection weights to the
  // output gate.
  l.PeepholeOutputGateWeight().randu();

  // For the peephole connection to give any output, we must also set the cell
  // value to something.
  arma::mat cell, storedCell;
  storedCell.randu(outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  cell = storedCell;

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // The updated cell is recalculated as half its previous value.  Then the
  // output gate value is the output peephole weights multiplied elementwise
  // with the updated cell; this is then passed through a sigmoid and multiplied
  // with the tanh of the updated cell.
  arma::mat expectedOutput = tanh(0.5 * storedCell) % (1.0 / (1.0 + exp(
      -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) %
      (0.5 * storedCell))));

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardNonRecurrentTest", "[ANNLayerTest]")
{
  // Make sure that when all recurrent weights are set to zero (including
  // peephole weights) that we get the same output as the first time step.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);
  l.CurrentStep(0); // So we will not be able to use the previous state.

  // Set all of the weights and recurrent state to random.
  l.Parameters().randu();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output1(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output1);

  // Now set the recurrent state to something random, and disable all recurrent
  // connections by setting their weights to zero.
  l.RecurrentState(l.PreviousStep()).randu();
  l.RecurrentState(l.CurrentStep()).randu();
  l.RecurrentBlockInputWeight().zeros();
  l.RecurrentInputGateWeight().zeros();
  l.RecurrentForgetGateWeight().zeros();
  l.RecurrentOutputGateWeight().zeros();
  l.PeepholeInputGateWeight().zeros();
  l.PeepholeForgetGateWeight().zeros();
  l.PeepholeOutputGateWeight().zeros();

  // Lastly, set the cell state for the last time step to 0.
  l.CurrentStep(1);
  arma::mat cell;
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  cell.zeros();

  arma::mat output2(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output2);

  REQUIRE(approx_equal(output1, output2, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardRecurrentOnlyTest", "[ANNLayerTest]")
{
  // Make sure that when all non-recurrent weights are set to zero (including
  // peephole weights) that we get the desired output.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);

  // Set the recurrent state to random values.
  l.CurrentStep(1);
  l.RecurrentState(l.PreviousStep()).randu();

  // Now set all the recurrent weights to something.
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  // For convenience in computing the expected output, make aliases of the
  // recurrent cell and output.
  arma::mat cell, y;
  MakeAlias(y, l.RecurrentState(l.PreviousStep()), outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);

  l.Forward(input, output);

  // This is manually computed and a little difficult to describe, but the
  // picture of the LSTM in Figure 1 of "LSTM: A Search Space Odyssey" should
  // make it fairly clear (with a little bit of effort).
  arma::mat expectedOutput =
      /* output gate */
      (1.0 / (1.0 + exp(-l.RecurrentOutputGateWeight() * y))) %
      /* cell after nonlinearity */
      tanh(
          /* input gate */
          (1.0 / (1.0 + exp(-l.RecurrentInputGateWeight() * y))) %
          /* block input */
          tanh(l.RecurrentBlockInputWeight() * y) +
          /* previous cell and forget gate */
          cell % (1.0 / (1.0 + exp(-l.RecurrentForgetGateWeight() * y))));

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardPeepholeOnlyTest", "[ANNLayerTest]")
{
  // Make sure that when all non-peephole weights are set to zero that we get
  // the desired output.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);

  // Set the recurrent state to random values.
  l.CurrentStep(1);
  l.RecurrentState(l.PreviousStep()).randu();

  // Now set all the peephole weights to something.
  l.PeepholeInputGateWeight().randu();
  l.PeepholeForgetGateWeight().randu();
  l.PeepholeOutputGateWeight().randu();
  // Also set the block input bias to something, otherwise the input gate gets
  // turned into zeros.
  l.BlockInputBias().ones();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  // For convenience in computing the expected output, make aliases of the
  // recurrent cell and output.
  arma::mat cell, y;
  MakeAlias(y, l.RecurrentState(l.PreviousStep()), outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);

  l.Forward(input, output);

  // This is manually computed and a little difficult to describe, but the
  // picture of the LSTM in Figure 1 of "LSTM: A Search Space Odyssey" should
  // make it fairly clear (with a little bit of effort).
  arma::mat newCell =
      /* input gate */
      (1.0 / (1.0 + exp(
          -repmat(l.PeepholeInputGateWeight(), 1, batchSize) % cell))) %
      /* block input */
      tanh(repmat(l.BlockInputBias(), 1, batchSize)) +
      /* forget gate multiplied by cell */
      cell % (1.0 / (1.0 + exp(
          -repmat(l.PeepholeForgetGateWeight(), 1, batchSize) % cell)));

  arma::mat expectedOutput =
      /* output gate */
      (1.0 / (1.0 + exp(
          -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % newCell))) %
      /* new cell after nonlinearity */
      tanh(newCell);

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMForwardBiasOnlyTest", "[ANNLayerTest]")
{
  // Make sure that we get the expected output when no inputs are given and only
  // the biases are set.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 2, weights);

  // Set the recurrent state to random values.
  l.CurrentStep(1);
  l.RecurrentState(l.PreviousStep()).randu();

  // Now set all the biases to something.
  l.BlockInputBias().randu();
  l.InputGateBias().randu();
  l.ForgetGateBias().randu();
  l.OutputGateBias().randu();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  // For convenience in computing the expected output, make aliases of the
  // recurrent cell.
  arma::mat cell;
  MakeAlias(cell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);

  // This is manually computed and a little difficult to describe, but the
  // picture of the LSTM in Figure 1 of "LSTM: A Search Space Odyssey" should
  // make it fairly clear (with a little bit of effort).
  arma::mat expectedOutput =
      /* output gate */
      (1.0 / (1.0 + exp(-repmat(l.OutputGateBias(), 1, batchSize)))) %
      /* cell after nonlinearity */
      tanh(
          /* block input */
          tanh(repmat(l.BlockInputBias(), 1, batchSize)) %
          /* input gate */
          (1.0 / (1.0 + exp(-repmat(l.InputGateBias(), 1, batchSize)))) +
          /* cell multiplied by forget gate */
          cell %
          (1.0 / (1.0 + exp(-repmat(l.ForgetGateBias(), 1, batchSize)))));

  REQUIRE(approx_equal(output, expectedOutput, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMBackwardInputOnlyTest", "[ANNLayerTest]")
{
  // Check that the LSTM backward pass is correct when all weights *except* the
  // input gate (and the block input bias) are zero.  This test is run twice;
  // once with random recurrent state but zero recurrent weights, and once with
  // random recurrent weights but at the last time step.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the input weights and the block input bias to something.
  l.InputGateWeight().randu();
  l.BlockInputBias().ones();

  // First test: we are at the last time step.  Set recurrent weights to
  // something random (they won't be used).
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  l.CurrentStep(0);

  arma::mat cell;
  MakeAlias(cell, l.RecurrentState(l.CurrentStep()), outputSize, batchSize,
      outputSize * batchSize);

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  l.CurrentStep(0, true);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dxIn(inputSize, batchSize, arma::fill::none);

  l.Backward(input, output, delta, dxIn);

  // Manually compute dout/dx.  Because there are no recurrent connections, the
  // math in Section B of "LSTM: A Search Space Odyssey" simplifies; only the
  // di_t term on dx_t is nonzero.
  arma::mat inputGate = (1.0 / (1.0 + exp(-l.InputGateWeight() * input)));
  arma::mat dxInExpected = l.InputGateWeight().t() *
      /* di_t = dc_t % z_t % sigmoid'(i_t) */
      /* dc_t = dy_t % o_t % (1 - (c_t)^2) */
      ((delta * 0.5 /* o_t */) % (1.0 - square(tanh(cell))) %
      tanh(repmat(l.BlockInputBias(), 1, batchSize)) %
      (inputGate % (1.0 - inputGate)));

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));

  // Second test: we are not the last step, but all recurrent weights are set to
  // 0.  This one is a little tricky: to correctly be "not in the last step", we
  // have to set the internally held workspace values correctly, by running
  // through a delta of zeros.
  arma::mat zeroDelta(outputSize, batchSize, arma::fill::zeros);
  l.RecurrentState(1).zeros();
  l.CurrentStep(1, true);
  l.Backward(input, output, zeroDelta, dxIn /* ignored */);

  l.CurrentStep(0, false);
  l.RecurrentBlockInputWeight().zeros();
  l.RecurrentInputGateWeight().zeros();
  l.RecurrentForgetGateWeight().zeros();
  l.RecurrentOutputGateWeight().zeros();

  l.Forward(input, output);
  l.Backward(input, output, delta, dxIn);

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMBackwardBlockInputOnlyTest", "[ANNLayerTest]")
{
  // Check that the LSTM backward pass is correct when all weights *except* the
  // block input are zero.  This test is run twice; once with random recurrent
  // state but zero recurrent weights, and once with random recurrent weights
  // but at the last time step.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the block input weights to something.
  l.BlockInputWeight().randu();

  // First test: we are at the last time step.  Set recurrent weights to
  // something random (they won't be used).
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  l.CurrentStep(0);

  arma::mat cell;
  MakeAlias(cell, l.RecurrentState(l.CurrentStep()), outputSize, batchSize,
      outputSize * batchSize);

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  l.CurrentStep(0, true);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dxIn(inputSize, batchSize, arma::fill::none);

  l.Backward(input, output, delta, dxIn);

  // Manually compute dout/dx.  Because there are no recurrent connections, the
  // math in Section B of "LSTM: A Search Space Odyssey" simplifies; only the
  // dz_t term on dx_t is nonzero.
  arma::mat dxInExpected = l.BlockInputWeight().t() *
      /* dz_t = dc_t % i_t % (1 - (z_t)^2) */
      /* dc_t = dy_t % o_t % (1 - (c_t)^2) */
      (((delta * 0.5 /* o_t */) % (1.0 - square(tanh(cell))) * 0.5) %
      (1.0 - square(tanh(l.BlockInputWeight() * input))));

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));

  // Second test: we are not the last step, but all recurrent weights are set to
  // 0.  This one is a little tricky: to correctly be "not in the last step", we
  // have to set the internally held workspace values correctly, by running
  // through a delta of zeros.
  arma::mat zeroDelta(outputSize, batchSize, arma::fill::zeros);
  l.RecurrentState(1).zeros();
  l.CurrentStep(1, true);
  l.Backward(input, output, zeroDelta, dxIn /* ignored */);

  l.CurrentStep(0, false);
  l.RecurrentBlockInputWeight().zeros();
  l.RecurrentInputGateWeight().zeros();
  l.RecurrentForgetGateWeight().zeros();
  l.RecurrentOutputGateWeight().zeros();

  l.Forward(input, output);
  l.Backward(input, output, delta, dxIn);

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMBackwardForgetOnlyTest", "[ANNLayerTest]")
{
  // Check that the LSTM backward pass is correct when all weights *except* the
  // forget gate weights are zero.  This test is run twice; once with random
  // recurrent state but zero recurrent weights, and once with random recurrent
  // weights but at the last time step.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the forget gate weights to something.
  l.ForgetGateWeight().randu();

  // First test: we are at the last time step.  Set recurrent weights to
  // something random (they won't be used).
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  l.CurrentStep(1, true);
  arma::mat cell, lastCell;
  MakeAlias(lastCell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell, l.RecurrentState(l.CurrentStep()), outputSize, batchSize,
      outputSize * batchSize);

  // Manually set the last cell to have some values, but the last output to be
  // zero.
  l.RecurrentState(l.PreviousStep()).zeros();
  lastCell.randu();

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);
  l.Forward(input, output);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dxIn(inputSize, batchSize, arma::fill::none);

  l.Backward(input, output, delta, dxIn);

  // Manually compute dout/dx.  Because there are no recurrent connections, the
  // math in Section B of "LSTM: A Search Space Odyssey" simplifies; only the
  // df_t term in dx_t is nonzero.
  arma::mat forgetGate = 1.0 / (1.0 + exp(-l.ForgetGateWeight() * input));
  arma::mat dxInExpected = l.ForgetGateWeight().t() *
      /* df_t = dc_t % c_{t - 1} % (f_t % (1 - f_t)) */
      /* dc_t = dy_t % o_t % (1 - (c_t)^2) */
      (((delta * 0.5 /* o_t */) % (1.0 - square(tanh(cell)))) %
      lastCell % (forgetGate % (1.0 - forgetGate)));

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));

  // Second test: we are not the last step, but all recurrent weights are set to
  // 0.  This one is a little tricky: to correctly be "not in the last step", we
  // have to set the internally held workspace values correctly, by running
  // through a delta of zeros.
  arma::mat zeroDelta(outputSize, batchSize, arma::fill::zeros);
  l.RecurrentState(2).zeros();
  l.CurrentStep(2, true);
  l.Backward(input, output, zeroDelta, dxIn /* ignored */);

  l.CurrentStep(1, false);
  l.RecurrentBlockInputWeight().zeros();
  l.RecurrentInputGateWeight().zeros();
  l.RecurrentForgetGateWeight().zeros();
  l.RecurrentOutputGateWeight().zeros();

  l.Forward(input, output);
  l.Backward(input, output, delta, dxIn);

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMBackwardOutputOnlyTest", "[ANNLayerTest]")
{
  // Check that the LSTM backward pass is correct when all weights *except* the
  // output gate weights are zero.  This test is run twice; once with random
  // recurrent state but zero recurrent weights, and once with random recurrent
  // weights but at the last time step.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the output gate weights to something.
  l.OutputGateWeight().randu();

  // First test: we are at the last time step.  Set recurrent weights to
  // something random (they won't be used).
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  l.CurrentStep(0);

  arma::mat cell;
  MakeAlias(cell, l.RecurrentState(l.CurrentStep()), outputSize, batchSize,
      outputSize * batchSize);

  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  l.Forward(input, output);

  l.CurrentStep(0, true);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dxIn(inputSize, batchSize, arma::fill::none);

  l.Backward(input, output, delta, dxIn);

  // Manually compute dout/dx.  Because there are no recurrent connections, the
  // math in Section B of "LSTM: A Search Space Odyssey" simplifies; only the
  // do_t term on dx_t is nonzero.
  arma::mat outputGate = 1.0 / (1.0 + exp(-l.OutputGateWeight() * input));
  arma::mat dxInExpected = l.OutputGateWeight().t() *
      /* do_t = dy_t % tanh(cell) % (o_t % (1 - o_t)) */
      (delta % tanh(cell) % (outputGate % (1.0 - outputGate)));

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));

  // Second test: we are not the last step, but all recurrent weights are set to
  // 0.  This one is a little tricky: to correctly be "not in the last step", we
  // have to set the internally held workspace values correctly, by running
  // through a delta of zeros.
  arma::mat zeroDelta(outputSize, batchSize, arma::fill::zeros);
  l.RecurrentState(1).zeros();
  l.CurrentStep(1, true);
  l.Backward(input, output, zeroDelta, dxIn /* ignored */);

  l.CurrentStep(0, false);
  l.RecurrentBlockInputWeight().zeros();
  l.RecurrentInputGateWeight().zeros();
  l.RecurrentForgetGateWeight().zeros();
  l.RecurrentOutputGateWeight().zeros();

  l.Forward(input, output);
  l.Backward(input, output, delta, dxIn);

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMBackwardNonRecurrentWeightsTest", "[ANNLayerTest]")
{
  // Ensure that the LSTM layer gives the correct results from Backward() when
  // all non-recurrent weights have values and all recurrent weights are zero.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the non-recurrent weights to something.
  l.BlockInputWeight().randu();
  l.InputGateWeight().randu();
  l.ForgetGateWeight().randu();
  l.OutputGateWeight().randu();

  // First test: we are at the last time step.  Set recurrent weights to
  // something random (they won't be used).
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  l.CurrentStep(0);

  // Initialize the recurrent state by passing random data through.
  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);
  l.Forward(input, output);

  l.CurrentStep(1, true);
  arma::mat cell, lastCell;
  MakeAlias(lastCell, l.RecurrentState(l.PreviousStep()), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell, l.RecurrentState(l.CurrentStep()), outputSize, batchSize,
      outputSize * batchSize);

  // Manually set the last cell to have some values, but the last output to be
  // zero.
  l.RecurrentState(l.PreviousStep()).zeros();
  lastCell.randu();

  l.Forward(input, output);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dxIn(inputSize, batchSize, arma::fill::none);

  l.Backward(input, output, delta, dxIn);

  // Manually compute dout/dx.  Because there are no recurrent connections, the
  // math in Section B of "LSTM: A Search Space Odyssey" simplifies, but not
  // really all that much.
  arma::mat outputGate = 1.0 / (1.0 + exp(-l.OutputGateWeight() * input));

  arma::mat dout = /* do_t = dy_t % h(c_t) % (o_t % (1 - o_t)) */
      delta % tanh(cell) % (outputGate % (1.0 - outputGate));

  arma::mat dc = /* dc_t = dy_t % o_t % h'(c_t) */
      delta % outputGate % (1.0 - square(tanh(cell)));

  arma::mat forgetGate = 1.0 / (1.0 + exp(-l.ForgetGateWeight() * input));

  arma::mat df = /* df_t = dc_t % c_{t - 1} % (f_t % (1 - f_t)) */
      dc % lastCell % (forgetGate % (1.0 - forgetGate));

  arma::mat inputGate = 1.0 / (1.0 + exp(-l.InputGateWeight() * input));
  arma::mat blockInput = tanh(l.BlockInputWeight() * input);

  arma::mat di = /* di_t = dc_t % z_t % (i_t % (1 - i_t)) */
      dc % blockInput % (inputGate % (1.0 - inputGate));

  arma::mat dz = /* dz_t = dc_t % i_t % (1.0 - square(z_t)) */
      dc % inputGate % (1.0 - square(blockInput));

  arma::mat dxInExpected = /* dx = W_z^T dz + W_i^T di + W_f^T df + W_o^T do */
      l.BlockInputWeight().t() * dz +
      l.InputGateWeight().t() * di +
      l.ForgetGateWeight().t() * df +
      l.OutputGateWeight().t() * dout;

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));

  // Second test: we are not the last step, but all recurrent weights are set to
  // 0.  This one is a little tricky: to correctly be "not in the last step", we
  // have to set the internally held workspace values correctly, by running
  // through a delta of zeros.
  arma::mat zeroDelta(outputSize, batchSize, arma::fill::zeros);
  l.RecurrentState(2).zeros();
  l.CurrentStep(2, true);
  l.Backward(input, output, zeroDelta, dxIn /* ignored */);

  l.CurrentStep(1, false);
  l.RecurrentBlockInputWeight().zeros();
  l.RecurrentInputGateWeight().zeros();
  l.RecurrentForgetGateWeight().zeros();
  l.RecurrentOutputGateWeight().zeros();

  l.Forward(input, output);
  l.Backward(input, output, delta, dxIn);

  REQUIRE(approx_equal(dxIn, dxInExpected, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMBackwardRecurrentWeightsTest", "[ANNLayerTest]")
{
  // Ensure that the LSTM layer gives the correct results from Backward() when
  // all recurrent weights (but not peephole weights) have values and all
  // recurrent weights are zero.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the recurrent weights to something.
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  // No matter what we do, the deltas should be zero because the non-recurrent
  // weight matrices are zero.
  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  l.Forward(input, output);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dx(inputSize, batchSize, arma::fill::none);

  l.Backward(input, output, delta, dx);

  REQUIRE(all(all(dx == 0.0)));

  l.CurrentStep(1);

  l.Backward(input, output, delta, dx);

  REQUIRE(all(all(dx == 0.0)));

  l.CurrentStep(0);

  l.Backward(input, output, delta, dx);

  REQUIRE(all(all(dx == 0.0)));
}

TEST_CASE("LSTMBackwardEachRecurrentWeightTest", "[ANNLayerTest]")
{
  // Isolate the components of the delta by setting the non-recurrent weights
  // individually to a random matrix, while holding the recurrent weights
  // constant.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the recurrent weights to something.
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentInputGateWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();

  // Construct the input and output that we will use.
  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  arma::mat cell, lastCell, y, lastY;
  MakeAlias(lastY, l.RecurrentState(1), outputSize, batchSize);
  MakeAlias(lastCell, l.RecurrentState(1), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(y, l.RecurrentState(2), outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(2), outputSize, batchSize,
      outputSize * batchSize);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dx(inputSize, batchSize, arma::fill::none);

  // Set the block input weight to something random so that we get output for
  // dx.
  l.BlockInputWeight().randu();

  // Now do the entire forward step sequence and then the first backward step.
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // With only W_z nonzero, dx = W_z^T dz.
  arma::mat outputGate = 1.0 /
      (1.0 + exp(-l.RecurrentOutputGateWeight() * lastY));

  arma::mat dc = /* dc_t = dy_t % o_t % h'(c_t) */
      delta % outputGate % (1.0 - square(tanh(cell)));

  arma::mat blockInput = tanh(l.BlockInputWeight() * input +
                              l.RecurrentBlockInputWeight() * lastY);
  arma::mat inputGate = 1.0 /
      (1.0 + exp(-l.RecurrentInputGateWeight() * lastY));
  arma::mat dz = /* dz_t = dc_t % i_t % (1 - z_t^2) */
      dc % inputGate % (1.0 - square(blockInput));

  arma::mat dxExpected = l.BlockInputWeight().t() * dz;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now take an additional time step back and just do a sanity check that this
  // changes the result.
  l.CurrentStep(1);
  l.Backward(input, output, delta, dx);
  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now isolate with W_i nonzero.
  l.BlockInputWeight().zeros();
  l.InputGateWeight().randu();

  // Now do the entire forward step sequence and then the first backward step.
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // Recompute all quantities.
  outputGate = 1.0 /
      (1.0 + exp(-l.RecurrentOutputGateWeight() * lastY));
  dc = /* dc_t = dy_t % o_t % h'(c_t) */
      delta % outputGate % (1.0 - square(tanh(cell)));
  blockInput = tanh(l.RecurrentBlockInputWeight() * lastY);
  inputGate = 1.0 / (1.0 + exp(-(l.InputGateWeight() * input +
                                 l.RecurrentInputGateWeight() * lastY)));
  arma::mat di = /* di_t = dc_t % z_t % (i_t % (1.0 - i_t)) */
      dc % blockInput % (inputGate % (1.0 - inputGate));

  dxExpected = l.InputGateWeight().t() * di;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now take an additional time step back and just do a sanity check that this
  // changes the result.
  l.CurrentStep(1);
  l.Backward(input, output, delta, dx);
  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now isolate with W_f nonzero.
  l.InputGateWeight().zeros();
  l.ForgetGateWeight().randu();

  // Do the entire forward step sequence and then the first backward step.
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // With only W_f nonzero, dx = W_f^T df.
  outputGate = 1.0 /
      (1.0 + exp(-l.RecurrentOutputGateWeight() * lastY));
  dc = /* dc_t = dy_t % o_t % h'(c_t) */
      delta % outputGate % (1.0 - square(tanh(cell)));
  arma::mat forgetGate = 1.0 / (1.0 + exp(-(l.ForgetGateWeight() * input +
      l.RecurrentForgetGateWeight() * lastY)));
  arma::mat df = dc % lastCell % (forgetGate % (1.0 - forgetGate));
  blockInput = tanh(l.RecurrentBlockInputWeight() * lastY);
  inputGate = 1.0 / (1.0 + exp(-(l.InputGateWeight() * input +
                                 l.RecurrentInputGateWeight() * lastY)));

  dxExpected = l.ForgetGateWeight().t() * df;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now take an additional time step back and just do a sanity check that this
  // changes the result.
  l.CurrentStep(1);
  l.Backward(input, output, delta, dx);
  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now isolate with W_o nonzero.
  l.ForgetGateWeight().zeros();
  l.OutputGateWeight().randu();

  // Do the entire forward step sequence and then the first backward step.
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // To compute do_t, we only need outputGate.
  outputGate = 1.0 /
      (1.0 + exp(-(l.OutputGateWeight() * input +
                   l.RecurrentOutputGateWeight() * lastY)));
  arma::mat dout = delta % tanh(cell) % (outputGate % (1.0 - outputGate));

  dxExpected = l.OutputGateWeight().t() * dout;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now take an additional time step back and just do a sanity check that this
  // changes the result.
  l.CurrentStep(1);
  l.Backward(input, output, delta, dx);
  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMBackwardPeepholeWeightsTest", "[ANNLayerTest]")
{
  // Isolate the components of the delta by setting the peephole weights
  // individually to a random matrix, while holding the recurrent weights
  // constant.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Now set the recurrent weights to something.
  l.PeepholeInputGateWeight().randu();
  l.PeepholeForgetGateWeight().randu();
  l.PeepholeOutputGateWeight().randu();

  // Construct the input and output that we will use.
  arma::mat input(inputSize, batchSize, arma::fill::randu);
  arma::mat output(outputSize, batchSize, arma::fill::none);

  arma::mat firstCell, cell, lastCell, y, lastY;
  MakeAlias(firstCell, l.RecurrentState(0), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(lastY, l.RecurrentState(1), outputSize, batchSize);
  MakeAlias(lastCell, l.RecurrentState(1), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(y, l.RecurrentState(2), outputSize, batchSize);
  MakeAlias(cell, l.RecurrentState(2), outputSize, batchSize,
      outputSize * batchSize);

  arma::mat delta(outputSize, batchSize, arma::fill::randu);
  arma::mat dx(inputSize, batchSize, arma::fill::none);

  // Set the block input weight to something random so that we get output for
  // dx.
  l.BlockInputWeight().randu();

  // Now do the entire forward step sequence and then the first backward step.
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // Now compute dx.  The only nonzero term will be dz.
  // dz_t = dc_t % i_t % (1 - (z_t)^2).
  // dc_t = dy_t % o_t % h'(c_t) + p_o % do_t + p_i % di_{t + 1} +
  //    p_f % df_{t + 1} + dc_{t + 1} % f_{t + 1}
  // dy_t = delta
  // o_t = p_o % c_t
  arma::mat outputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % cell));
  arma::mat dout = delta % tanh(cell) % (outputGate % (1.0 - outputGate));
  arma::mat dc = delta % outputGate % (1.0 - square(tanh(cell))) +
      repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % dout;
      // Time steps t+1 aren't valid.
  arma::mat inputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeInputGateWeight(), 1, batchSize) % lastCell));
  arma::mat blockInput = tanh(l.BlockInputWeight() * input);
  arma::mat dz = dc % inputGate % (1.0 - square(blockInput));

  arma::mat dxExpected = l.BlockInputWeight().t() * dz;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now step back a time step and make sure the output has changed.
  l.CurrentStep(1);
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Next isolate the input gate.  To get any result for dx, we need to set the
  // block input bias to a nonzero value.
  l.BlockInputWeight().zeros();
  l.InputGateWeight().randu();
  l.BlockInputBias().ones();

  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // Compute dx.  The only nonzero term will be di.
  // di_t = dc_t % z_t % (i_t % (1.0 - i_t))
  inputGate = 1.0 / (1.0 + exp(-(l.InputGateWeight() * input +
      repmat(l.PeepholeInputGateWeight(), 1, batchSize) % lastCell)));
  outputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % cell));
  dout = delta % tanh(cell) % (outputGate % (1.0 - outputGate));
  dc = delta % outputGate % (1.0 - square(tanh(cell))) +
      repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % dout;
  arma::mat di = (dc * tanh(1.0)) % (inputGate % (1.0 - inputGate));

  dxExpected = l.InputGateWeight().t() * di;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now step back a time step and make sure the output has changed.
  l.CurrentStep(1);
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now isolate with W_f nonzero.
  l.InputGateWeight().zeros();
  l.BlockInputBias().zeros();
  l.ForgetGateWeight().randu();

  // Do the entire forward step sequence and then the first backward step.
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // Compute dx.  The only nonzero term will be df.
  // df_t = dc_t % c_{t - 1} % (f_t % (1.0 - f_t))
  arma::mat forgetGate = 1.0 / (1.0 + exp(-(l.ForgetGateWeight() * input +
      repmat(l.PeepholeForgetGateWeight(), 1, batchSize) % lastCell)));
  outputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % cell));
  dout = delta % tanh(cell) % (outputGate % (1.0 - outputGate));
  dc = delta % outputGate % (1.0 - square(tanh(cell))) +
      repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % dout;
  arma::mat df = dc % lastCell % (forgetGate % (1.0 - forgetGate));

  dxExpected = l.ForgetGateWeight().t() * df;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now step back a time step and make sure the output has changed.
  l.CurrentStep(1);
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Lastly, isolate with W_o nonzero.
  l.ForgetGateWeight().zeros();
  l.OutputGateWeight().randu();

  // Do the entire forward step sequence and then the first backward step.
  l.CurrentStep(0);
  l.Forward(input, output);
  l.CurrentStep(1);
  l.Forward(input, output);
  l.CurrentStep(2, true);
  // Set random internal state, in case it was 0.
  lastY.randu();
  lastCell.randu();
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  // Now the only nonzero term in dx is do_t.
  // do_t = delta % tanh(cell) % (o_t % (1.0 - o_t))
  outputGate = 1.0 / (1.0 + exp(-(l.OutputGateWeight() * input +
      repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % cell)));
  dout = delta % tanh(cell) % (outputGate % (1.0 - outputGate));

  dxExpected = l.OutputGateWeight().t() * dout;

  REQUIRE(approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));

  // Now step back a time step and make sure the output has changed.
  l.CurrentStep(1);
  l.Forward(input, output);
  l.Backward(input, output, delta, dx);

  REQUIRE(!approx_equal(dx, dxExpected, "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMGradientNonRecurrentWeightsTest", "[ANNLayerTest]")
{
  // Set the non-recurrent weights of an LSTM cell to random values, then
  // compute the gradients of the weights.  We will do this for three time
  // steps.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  // Set non-recurrent weights to nonzero values.
  l.InputGateWeight().randu();
  l.BlockInputWeight().randu();
  l.ForgetGateWeight().randu();
  l.OutputGateWeight().randu();

  arma::mat input0(inputSize, batchSize, arma::fill::randu);
  arma::mat input1(inputSize, batchSize, arma::fill::randu);
  arma::mat input2(inputSize, batchSize, arma::fill::randu);
  arma::mat output0(outputSize, batchSize, arma::fill::none);
  arma::mat output1(outputSize, batchSize, arma::fill::none);
  arma::mat output2(outputSize, batchSize, arma::fill::none);

  // Create aliases to internal state.
  arma::mat cell0, cell1, cell2;
  MakeAlias(cell0, l.RecurrentState(0), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell1, l.RecurrentState(1), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell2, l.RecurrentState(2), outputSize, batchSize,
      outputSize * batchSize);

  // Now take the forward steps.
  l.CurrentStep(0);
  l.Forward(input0, output0);
  l.CurrentStep(1);
  l.Forward(input1, output1);
  l.CurrentStep(2, true);
  l.Forward(input2, output2);

  arma::mat dx(inputSize, batchSize, arma::fill::zeros);
  arma::mat gradient(l.WeightSize(), 1, arma::fill::zeros);
  arma::mat delta0(outputSize, batchSize, arma::fill::randu);
  arma::mat delta1(outputSize, batchSize, arma::fill::randu);
  arma::mat delta2(outputSize, batchSize, arma::fill::randu);

  // Create aliases for gradient terms.
  arma::mat inputGateWeightGrad, blockInputWeightGrad, forgetGateWeightGrad,
      outputGateWeightGrad;
  MakeAlias(blockInputWeightGrad, gradient, outputSize, inputSize);
  MakeAlias(inputGateWeightGrad, gradient, outputSize, inputSize,
      outputSize * inputSize);
  MakeAlias(forgetGateWeightGrad, gradient, outputSize, inputSize,
      2 * outputSize * inputSize);
  MakeAlias(outputGateWeightGrad, gradient, outputSize, inputSize,
      3 * outputSize * inputSize);

  const size_t biasOffset = 4 * outputSize * inputSize;
  arma::mat inputGateBiasGrad, blockInputBiasGrad, forgetGateBiasGrad,
      outputGateBiasGrad;
  MakeAlias(blockInputBiasGrad, gradient, outputSize, 1, biasOffset);
  MakeAlias(inputGateBiasGrad, gradient, outputSize, 1,
      biasOffset + outputSize);
  MakeAlias(forgetGateBiasGrad, gradient, outputSize, 1,
      biasOffset + 2 * outputSize);
  MakeAlias(outputGateBiasGrad, gradient, outputSize, 1,
      biasOffset + 3 * outputSize);

  arma::mat recurrentBlockInputWeightGrad, recurrentInputGateWeightGrad,
      recurrentForgetGateWeightGrad, recurrentOutputGateWeightGrad;
  const size_t recurrentOffset = biasOffset + 4 * outputSize;
  MakeAlias(recurrentBlockInputWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset);
  MakeAlias(recurrentInputGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + outputSize * outputSize);
  MakeAlias(recurrentForgetGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + 2 * outputSize * outputSize);
  MakeAlias(recurrentOutputGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + 3 * outputSize * outputSize);

  // Now take the backward and gradient pass.
  l.Backward(input2, output2, delta2, dx);
  l.Gradient(input2, dx, gradient);

  // Compute the true values of the gradient.  We need all of the backward
  // values for this.
  arma::mat inputGate = 1.0 / (1.0 + exp(-l.InputGateWeight() * input2));
  arma::mat blockInput = tanh(l.BlockInputWeight() * input2);
  arma::mat forgetGate = 1.0 / (1.0 + exp(-l.ForgetGateWeight() * input2));
  arma::mat outputGate = 1.0 / (1.0 + exp(-l.OutputGateWeight() * input2));
  arma::mat dout = delta2 % tanh(cell2) % (outputGate % (1.0 - outputGate));
  arma::mat dc = delta2 % outputGate % (1.0 - square(tanh(cell2)));
  arma::mat df = dc % cell1 % (forgetGate % (1.0 - forgetGate));
  arma::mat di = dc % blockInput % (inputGate % (1.0 - inputGate));
  arma::mat dz = dc % inputGate % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  arma::mat dwInputGate = di * input2.t();
  arma::mat dwBlockInput = dz * input2.t();
  arma::mat dwForgetGate = df * input2.t();
  arma::mat dwOutputGate = dout * input2.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  // dR_* = <d*_{t + 1}, y_t>
  // However, at the last time step, there are no values for d*_{t + 1}, so all
  // of these gradients are 0.
  REQUIRE(all(all(recurrentInputGateWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentBlockInputWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentForgetGateWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentOutputGateWeightGrad == 0.0)));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));

  // Now take a time step backwards.
  l.CurrentStep(1);
  l.Backward(input1, output1, delta1, dx);
  l.Gradient(input1, dx, gradient);

  // Precompute recurrent weight gradients, as they depend on the deltas from
  // the last time step.
  // dR_* = <d*_{t + 1}, y_t>
  arma::mat drInputGate = di * output1.t();
  arma::mat drBlockInput = dz * output1.t();
  arma::mat drForgetGate = df * output1.t();
  arma::mat drOutputGate = dout * output1.t();

  // Recompute the true values; the equations here are mostly the same, but that
  // is because the recurrent connections are all zero (so no information from
  // time step 2 is propagated).
  inputGate = 1.0 / (1.0 + exp(-l.InputGateWeight() * input1));
  blockInput = tanh(l.BlockInputWeight() * input1);
  outputGate = 1.0 / (1.0 + exp(-l.OutputGateWeight() * input1));
  dout = delta1 % tanh(cell1) % (outputGate % (1.0 - outputGate));
  dc = delta1 % outputGate % (1.0 - square(tanh(cell1))) +
      dc % forgetGate; // note these refer to dc_2 and f_2.
  forgetGate = 1.0 / (1.0 + exp(-l.ForgetGateWeight() * input1));
  df = dc % cell0 % (forgetGate % (1.0 - forgetGate));
  di = dc % blockInput % (inputGate % (1.0 - inputGate));
  dz = dc % inputGate % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  dwInputGate = di * input1.t();
  dwBlockInput = dz * input1.t();
  dwForgetGate = df * input1.t();
  dwOutputGate = dout * input1.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  REQUIRE(approx_equal(recurrentInputGateWeightGrad, drInputGate, "both", 1e-5,
      1e-5));
  REQUIRE(approx_equal(recurrentBlockInputWeightGrad, drBlockInput, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentForgetGateWeightGrad, drForgetGate, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentOutputGateWeightGrad, drOutputGate, "both",
      1e-5, 1e-5));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));

  // Now do the first time step.
  l.CurrentStep(0);
  l.Backward(input0, output0, delta0, dx);
  l.Gradient(input0, dx, gradient);

  // Precompute recurrent weight gradients, as they depend on the deltas from
  // the last time step.
  // dR_* = <d*_{t + 1}, y_t>
  drInputGate = di * output0.t();
  drBlockInput = dz * output0.t();
  drForgetGate = df * output0.t();
  drOutputGate = dout * output0.t();

  // Recompute the true values; these are now slightly different because we are
  // at the first step (so c_{t - 1} is zeros).
  inputGate = 1.0 / (1.0 + exp(-l.InputGateWeight() * input0));
  blockInput = tanh(l.BlockInputWeight() * input0);
  outputGate = 1.0 / (1.0 + exp(-l.OutputGateWeight() * input0));
  dout = delta0 % tanh(cell0) % (outputGate % (1.0 - outputGate));
  dc = delta0 % outputGate % (1.0 - square(tanh(cell0))) +
      dc % forgetGate; // Note that these refer to dc_1 and f_1.
  df.zeros();
  di = dc % blockInput % (inputGate % (1.0 - inputGate));
  dz = dc % inputGate % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  dwInputGate = di * input0.t();
  dwBlockInput = dz * input0.t();
  dwForgetGate = df * input0.t();
  dwOutputGate = dout * input0.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  REQUIRE(approx_equal(recurrentInputGateWeightGrad, drInputGate, "both", 1e-5,
      1e-5));
  REQUIRE(approx_equal(recurrentBlockInputWeightGrad, drBlockInput, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentForgetGateWeightGrad, drForgetGate, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentOutputGateWeightGrad, drOutputGate, "both",
      1e-5, 1e-5));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMGradientRecurrentWeightsTest", "[ANNLayerTest]")
{
  // Set only the recurrent weights of an LSTM cell to random values, then
  // compute the gradients of the weights.  We will do this for three time
  // steps.
  const size_t inputSize = 16;
  const size_t batchSize = 32;
  const size_t outputSize = 10;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  l.RecurrentInputGateWeight().randu();
  l.RecurrentBlockInputWeight().randu();
  l.RecurrentForgetGateWeight().randu();
  l.RecurrentOutputGateWeight().randu();
  // We also set the block input bias so that the cell state is not always 0.
  // (Otherwise there is no way for data to get into the cell.)
  l.BlockInputBias().randu();

  arma::mat input0(inputSize, batchSize, arma::fill::randu);
  arma::mat input1(inputSize, batchSize, arma::fill::randu);
  arma::mat input2(inputSize, batchSize, arma::fill::randu);
  arma::mat output0(outputSize, batchSize, arma::fill::none);
  arma::mat output1(outputSize, batchSize, arma::fill::none);
  arma::mat output2(outputSize, batchSize, arma::fill::none);

  // Create aliases to internal state.
  arma::mat cell0, cell1, cell2;
  MakeAlias(cell0, l.RecurrentState(0), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell1, l.RecurrentState(1), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell2, l.RecurrentState(2), outputSize, batchSize,
      outputSize * batchSize);

  // Now take the forward steps.
  l.CurrentStep(0);
  l.Forward(input0, output0);
  l.CurrentStep(1);
  l.Forward(input1, output1);
  l.CurrentStep(2, true);
  l.Forward(input2, output2);

  arma::mat dx(inputSize, batchSize, arma::fill::zeros);
  arma::mat gradient(l.WeightSize(), 1, arma::fill::zeros);
  arma::mat delta0(outputSize, batchSize, arma::fill::randu);
  arma::mat delta1(outputSize, batchSize, arma::fill::randu);
  arma::mat delta2(outputSize, batchSize, arma::fill::randu);

  // Create aliases for gradient terms.
  arma::mat inputGateWeightGrad, blockInputWeightGrad, forgetGateWeightGrad,
      outputGateWeightGrad;
  MakeAlias(blockInputWeightGrad, gradient, outputSize, inputSize);
  MakeAlias(inputGateWeightGrad, gradient, outputSize, inputSize,
      outputSize * inputSize);
  MakeAlias(forgetGateWeightGrad, gradient, outputSize, inputSize,
      2 * outputSize * inputSize);
  MakeAlias(outputGateWeightGrad, gradient, outputSize, inputSize,
      3 * outputSize * inputSize);

  const size_t biasOffset = 4 * outputSize * inputSize;
  arma::mat inputGateBiasGrad, blockInputBiasGrad, forgetGateBiasGrad,
      outputGateBiasGrad;
  MakeAlias(blockInputBiasGrad, gradient, outputSize, 1, biasOffset);
  MakeAlias(inputGateBiasGrad, gradient, outputSize, 1,
      biasOffset + outputSize);
  MakeAlias(forgetGateBiasGrad, gradient, outputSize, 1,
      biasOffset + 2 * outputSize);
  MakeAlias(outputGateBiasGrad, gradient, outputSize, 1,
      biasOffset + 3 * outputSize);

  arma::mat recurrentBlockInputWeightGrad, recurrentInputGateWeightGrad,
      recurrentForgetGateWeightGrad, recurrentOutputGateWeightGrad;
  const size_t recurrentOffset = biasOffset + 4 * outputSize;
  MakeAlias(recurrentBlockInputWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset);
  MakeAlias(recurrentInputGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + outputSize * outputSize);
  MakeAlias(recurrentForgetGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + 2 * outputSize * outputSize);
  MakeAlias(recurrentOutputGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + 3 * outputSize * outputSize);

  // Now take the backward and gradient pass.
  l.Backward(input2, output2, delta2, dx);
  l.Gradient(input2, dx, gradient);

  // Compute the true values of the gradient.  We need all of the backward
  // values for this.
  arma::mat dy = delta2;
  arma::mat outputGate = 1.0 / (1.0 +
      exp(-l.RecurrentOutputGateWeight() * output1));
  arma::mat dout = dy % tanh(cell2) % (outputGate % (1.0 - outputGate));
  arma::mat dc = dy % outputGate % (1.0 - square(tanh(cell2)));
  arma::mat forgetGate = 1.0 / (1.0 +
      exp(-l.RecurrentForgetGateWeight() * output1));
  arma::mat df = dc % cell1 % (forgetGate % (1.0 - forgetGate));
  arma::mat inputGate = 1.0 / (1.0 +
      exp(-l.RecurrentInputGateWeight() * output1));
  arma::mat blockInput = tanh(l.RecurrentBlockInputWeight() * output1 +
      repmat(l.BlockInputBias(), 1, batchSize));
  arma::mat di = dc % blockInput % (inputGate % (1.0 - inputGate));
  arma::mat dz = dc % inputGate % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  arma::mat dwInputGate = di * input2.t();
  arma::mat dwBlockInput = dz * input2.t();
  arma::mat dwForgetGate = df * input2.t();
  arma::mat dwOutputGate = dout * input2.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  // dR_* = <d*_{t + 1}, y_t>
  // However, at the last time step, there are no values for d*_{t + 1}, so all
  // of these gradients are 0.
  REQUIRE(all(all(recurrentInputGateWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentBlockInputWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentForgetGateWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentOutputGateWeightGrad == 0.0)));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));

  // Now take a time step backwards.
  l.CurrentStep(1);
  l.Backward(input1, output1, delta1, dx);
  l.Gradient(input1, dx, gradient);

  // Precompute recurrent weight gradients, as they depend on the deltas from
  // the last time step.
  // dR_* = <d*_{t + 1}, y_t>
  arma::mat drInputGate = di * output1.t();
  arma::mat drBlockInput = dz * output1.t();
  arma::mat drForgetGate = df * output1.t();
  arma::mat drOutputGate = dout * output1.t();

  // Recompute the true values.
  dy = delta1 + l.RecurrentBlockInputWeight().t() * dz +
      l.RecurrentInputGateWeight().t() * di +
      l.RecurrentForgetGateWeight().t() * df +
      l.RecurrentOutputGateWeight().t() * dout;
  outputGate = 1.0 / (1.0 +
      exp(-l.RecurrentOutputGateWeight() * output0));
  dout = dy % tanh(cell1) % (outputGate % (1.0 - outputGate));
  dc = dy % outputGate % (1.0 - square(tanh(cell1))) +
      dc % forgetGate; // Note here these refer to dc_2 and f_2.
  forgetGate = 1.0 / (1.0 +
      exp(-l.RecurrentForgetGateWeight() * output0));
  df = dc % cell0 % (forgetGate % (1.0 - forgetGate));
  inputGate = 1.0 / (1.0 +
      exp(-l.RecurrentInputGateWeight() * output0));
  blockInput = tanh(l.RecurrentBlockInputWeight() * output0 +
      repmat(l.BlockInputBias(), 1, batchSize));
  di = dc % blockInput % (inputGate % (1.0 - inputGate));
  dz = dc % inputGate % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  dwInputGate = di * input1.t();
  dwBlockInput = dz * input1.t();
  dwForgetGate = df * input1.t();
  dwOutputGate = dout * input1.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  REQUIRE(approx_equal(recurrentInputGateWeightGrad, drInputGate, "both", 1e-5,
      1e-5));
  REQUIRE(approx_equal(recurrentBlockInputWeightGrad, drBlockInput, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentForgetGateWeightGrad, drForgetGate, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentOutputGateWeightGrad, drOutputGate, "both",
      1e-5, 1e-5));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));

  // Now do the first time step.
  l.CurrentStep(0);
  l.Backward(input0, output0, delta0, dx);
  l.Gradient(input0, dx, gradient);

  // Precompute recurrent weight gradients, as they depend on the deltas from
  // the last time step.
  // dR_* = <d*_{t + 1}, y_t>
  drInputGate = di * output0.t();
  drBlockInput = dz * output0.t();
  drForgetGate = df * output0.t();
  drOutputGate = dout * output0.t();

  // Recompute the true values.
  dy = delta0 + l.RecurrentBlockInputWeight().t() * dz +
      l.RecurrentInputGateWeight().t() * di +
      l.RecurrentForgetGateWeight().t() * df +
      l.RecurrentOutputGateWeight().t() * dout;
  outputGate.fill(0.5); // This is the sigmoid of all zeros.
  dout = dy % tanh(cell0) % (outputGate % (1.0 - outputGate));
  dc = dy % outputGate % (1.0 - square(tanh(cell0))) +
      dc % forgetGate; // Note here these refer to dc_1 and f_1.
  forgetGate.fill(0.5);
  df.zeros();
  blockInput = tanh(repmat(l.BlockInputBias(), 1, batchSize));
  di = (dc % blockInput) * 0.25;
  dz = (dc * 0.5) % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  dwInputGate = di * input0.t();
  dwBlockInput = dz * input0.t();
  dwForgetGate = df * input0.t();
  dwOutputGate = dout * input0.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  REQUIRE(approx_equal(recurrentInputGateWeightGrad, drInputGate, "both", 1e-5,
      1e-5));
  REQUIRE(approx_equal(recurrentBlockInputWeightGrad, drBlockInput, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentForgetGateWeightGrad, drForgetGate, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentOutputGateWeightGrad, drOutputGate, "both",
      1e-5, 1e-5));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));
}

TEST_CASE("LSTMGradientPeepholeWeightsTest", "[ANNLayerTest]")
{
  // Set only the peephole weights of an LSTM cell to random values, then
  // compute the gradients of the weights.  We will do this for three time
  // steps.
  const size_t inputSize = 3;
  const size_t batchSize = 5;
  const size_t outputSize = 4;
  arma::mat weights;
  LSTM l = SetupLSTM(inputSize, batchSize, outputSize, 3, weights);

  l.PeepholeInputGateWeight().randu();
  l.PeepholeForgetGateWeight().randu();
  l.PeepholeOutputGateWeight().randu();
  // Here we will also set the block input bias to 1, so that z_t is not always
  // 0 and so that the cell state is not always zero.
  l.BlockInputBias().randu();

  arma::mat input0(inputSize, batchSize, arma::fill::randu);
  arma::mat input1(inputSize, batchSize, arma::fill::randu);
  arma::mat input2(inputSize, batchSize, arma::fill::randu);
  arma::mat output0(outputSize, batchSize, arma::fill::none);
  arma::mat output1(outputSize, batchSize, arma::fill::none);
  arma::mat output2(outputSize, batchSize, arma::fill::none);

  // Create aliases to internal state.
  arma::mat cell0, cell1, cell2;
  MakeAlias(cell0, l.RecurrentState(0), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell1, l.RecurrentState(1), outputSize, batchSize,
      outputSize * batchSize);
  MakeAlias(cell2, l.RecurrentState(2), outputSize, batchSize,
      outputSize * batchSize);

  // Now take the forward steps.
  l.CurrentStep(0);
  l.Forward(input0, output0);
  l.CurrentStep(1);
  l.Forward(input1, output1);
  l.CurrentStep(2, true);
  l.Forward(input2, output2);

  arma::mat dx(inputSize, batchSize, arma::fill::zeros);
  arma::mat gradient(l.WeightSize(), 1, arma::fill::zeros);
  arma::mat delta0(outputSize, batchSize, arma::fill::randu);
  arma::mat delta1(outputSize, batchSize, arma::fill::randu);
  arma::mat delta2(outputSize, batchSize, arma::fill::randu);

  // Create aliases for gradient terms.
  arma::mat inputGateWeightGrad, blockInputWeightGrad, forgetGateWeightGrad,
      outputGateWeightGrad;
  MakeAlias(blockInputWeightGrad, gradient, outputSize, inputSize);
  MakeAlias(inputGateWeightGrad, gradient, outputSize, inputSize,
      outputSize * inputSize);
  MakeAlias(forgetGateWeightGrad, gradient, outputSize, inputSize,
      2 * outputSize * inputSize);
  MakeAlias(outputGateWeightGrad, gradient, outputSize, inputSize,
      3 * outputSize * inputSize);

  const size_t biasOffset = 4 * outputSize * inputSize;
  arma::mat inputGateBiasGrad, blockInputBiasGrad, forgetGateBiasGrad,
      outputGateBiasGrad;
  MakeAlias(blockInputBiasGrad, gradient, outputSize, 1, biasOffset);
  MakeAlias(inputGateBiasGrad, gradient, outputSize, 1,
      biasOffset + outputSize);
  MakeAlias(forgetGateBiasGrad, gradient, outputSize, 1,
      biasOffset + 2 * outputSize);
  MakeAlias(outputGateBiasGrad, gradient, outputSize, 1,
      biasOffset + 3 * outputSize);

  arma::mat recurrentBlockInputWeightGrad, recurrentInputGateWeightGrad,
      recurrentForgetGateWeightGrad, recurrentOutputGateWeightGrad;
  const size_t recurrentOffset = biasOffset + 4 * outputSize;
  MakeAlias(recurrentBlockInputWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset);
  MakeAlias(recurrentInputGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + outputSize * outputSize);
  MakeAlias(recurrentForgetGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + 2 * outputSize * outputSize);
  MakeAlias(recurrentOutputGateWeightGrad, gradient, outputSize, outputSize,
      recurrentOffset + 3 * outputSize * outputSize);

  // Now take the backward and gradient pass.
  l.Backward(input2, output2, delta2, dx);
  l.Gradient(input2, dx, gradient);

  // Compute the true values of the gradient.  We need all of the backward
  // values for this.
  arma::mat outputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % cell2));
  arma::mat dout = delta2 % tanh(cell2) % (outputGate % (1.0 - outputGate));
  arma::mat dc = delta2 % outputGate % (1.0 - square(tanh(cell2))) +
      repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % dout;
  arma::mat forgetGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeForgetGateWeight(), 1, batchSize) % cell1));
  arma::mat df = dc % cell1 % (forgetGate % (1.0 - forgetGate));
  arma::mat inputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeInputGateWeight(), 1, batchSize) % cell1));
  arma::mat blockInput = tanh(repmat(l.BlockInputBias(), 1, batchSize));
  arma::mat di = dc % blockInput % (inputGate % (1.0 - inputGate));
  arma::mat dz = dc % inputGate % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  arma::mat dwInputGate = di * input2.t();
  arma::mat dwBlockInput = dz * input2.t();
  arma::mat dwForgetGate = df * input2.t();
  arma::mat dwOutputGate = dout * input2.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  // dR_* = <d*_{t + 1}, y_t>
  // However, at the last time step, there are no values for d*_{t + 1}, so all
  // of these gradients are 0.
  REQUIRE(all(all(recurrentInputGateWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentBlockInputWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentForgetGateWeightGrad == 0.0)));
  REQUIRE(all(all(recurrentOutputGateWeightGrad == 0.0)));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));

  // Now take a time step backwards.
  l.CurrentStep(1);
  l.Backward(input1, output1, delta1, dx);
  l.Gradient(input1, dx, gradient);

  // Precompute recurrent weight gradients, as they depend on the deltas from
  // the last time step.
  // dR_* = <d*_{t + 1}, y_t>
  arma::mat drInputGate = di * output1.t();
  arma::mat drBlockInput = dz * output1.t();
  arma::mat drForgetGate = df * output1.t();
  arma::mat drOutputGate = dout * output1.t();

  // Recompute the true values.
  outputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % cell1));
  dout = delta1 % tanh(cell1) % (outputGate % (1.0 - outputGate));
  dc = delta1 % outputGate % (1.0 - square(tanh(cell1))) +
      repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % dout +
      repmat(l.PeepholeForgetGateWeight(), 1, batchSize) % df +
      repmat(l.PeepholeInputGateWeight(), 1, batchSize) % di +
      dc % forgetGate; // Note that these refer to dc_2 and f_2.
  forgetGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeForgetGateWeight(), 1, batchSize) % cell0));
  df = dc % cell0 % (forgetGate % (1.0 - forgetGate));
  inputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeInputGateWeight(), 1, batchSize) % cell0));
  di = dc % blockInput % (inputGate % (1.0 - inputGate));
  dz = dc % inputGate % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  dwInputGate = di * input1.t();
  dwBlockInput = dz * input1.t();
  dwForgetGate = df * input1.t();
  dwOutputGate = dout * input1.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  REQUIRE(approx_equal(recurrentInputGateWeightGrad, drInputGate, "both", 1e-5,
      1e-5));
  REQUIRE(approx_equal(recurrentBlockInputWeightGrad, drBlockInput, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentForgetGateWeightGrad, drForgetGate, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentOutputGateWeightGrad, drOutputGate, "both",
      1e-5, 1e-5));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));

  // Now do the first time step.
  l.CurrentStep(0);
  l.Backward(input0, output0, delta0, dx);
  l.Gradient(input0, dx, gradient);

  // Precompute recurrent weight gradients, as they depend on the deltas from
  // the last time step.
  // dR_* = <d*_{t + 1}, y_t>
  drInputGate = di * output0.t();
  drBlockInput = dz * output0.t();
  drForgetGate = df * output0.t();
  drOutputGate = dout * output0.t();

  // Recompute the true values.
  outputGate = 1.0 / (1.0 + exp(
      -repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % cell0));
  dout = delta0 % tanh(cell0) % (outputGate % (1.0 - outputGate));
  dc = delta0 % outputGate % (1.0 - square(tanh(cell0))) +
      repmat(l.PeepholeOutputGateWeight(), 1, batchSize) % dout +
      repmat(l.PeepholeForgetGateWeight(), 1, batchSize) % df +
      repmat(l.PeepholeInputGateWeight(), 1, batchSize) % di +
      dc % forgetGate; // Note that these refer to dc_1 and f_1.
  df.zeros();
  di = (dc % blockInput) * 0.25;
  dz = (dc * 0.5) % (1.0 - square(blockInput));

  // dW_* = < d*, x_t >
  dwInputGate = di * input0.t();
  dwBlockInput = dz * input0.t();
  dwForgetGate = df * input0.t();
  dwOutputGate = dout * input0.t();

  REQUIRE(approx_equal(inputGateWeightGrad, dwInputGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputWeightGrad, dwBlockInput, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateWeightGrad, dwForgetGate, "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateWeightGrad, dwOutputGate, "both", 1e-5, 1e-5));

  REQUIRE(approx_equal(recurrentInputGateWeightGrad, drInputGate, "both", 1e-5,
      1e-5));
  REQUIRE(approx_equal(recurrentBlockInputWeightGrad, drBlockInput, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentForgetGateWeightGrad, drForgetGate, "both",
      1e-5, 1e-5));
  REQUIRE(approx_equal(recurrentOutputGateWeightGrad, drOutputGate, "both",
      1e-5, 1e-5));

  // db_* = d*_t (sum across the whole batch)
  REQUIRE(approx_equal(inputGateBiasGrad, sum(di, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(blockInputBiasGrad, sum(dz, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(forgetGateBiasGrad, sum(df, 1), "both", 1e-5, 1e-5));
  REQUIRE(approx_equal(outputGateBiasGrad, sum(dout, 1), "both", 1e-5, 1e-5));
}
