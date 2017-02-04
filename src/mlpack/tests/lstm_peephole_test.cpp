/**
 * @file lstm_peephole_test.cpp
 * @author Marcus Edel
 *
 * Tests the LSTM peepholes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/lstm_layer.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;


BOOST_AUTO_TEST_SUITE(LSTMPeepholeTest);

/*
 * Test the peephole connections in the forward pass. The test is a modification
 * of the peephole test originally written by Tom Schaul.
 */
BOOST_AUTO_TEST_CASE(LSTMPeepholeForwardTest)
{
  double state1 = 0.2;
  double state2 = 0.345;
  double state3 = -0.135;
  double state4 = 10000;

  arma::colvec input, output;

  LSTMLayer<> hiddenLayer0(1, 6, true);

  hiddenLayer0.InGatePeepholeWeights() = arma::mat("3");
  hiddenLayer0.ForgetGatePeepholeWeights() = arma::mat("4");
  hiddenLayer0.OutGatePeepholeWeights() = arma::mat("5");

  // Set the LSTM state to state1 (state = inGateActivation * cellActivation
  // = 1 / (1 + e^(-1000)) * tanh(atanh(0.2)) = 1 * 0.2 = 0.2).
  // outputActivation = outGateActivation * stateActivation
  // = tanh((0.2)) * (1 / (1 + e^1000)) = 0.
  input << state4 << state4 << std::atanh(state1) << -state4;
  hiddenLayer0.FeedForward(input, output);
  BOOST_REQUIRE_CLOSE(output(0), 0, 1e-3);

  // Verify that the LSTM state is correctly stored.
  input.clear();
  input << -state4 << state4 << state4 << state4;
  hiddenLayer0.FeedForward(input, output);
  BOOST_REQUIRE_CLOSE(output(0), std::tanh(state1), 1e-3);

  // Add state2 to the LSTM state.
  // state = state + forgateGateActivation * state(t - 1) = 0.345 + 1 * 0.2 =
  // 0.545
  input.clear();
  input << state4 << state4 << std::atanh(state2) << state4;
  hiddenLayer0.FeedForward(input, output);
  BOOST_REQUIRE_CLOSE(output(0), std::tanh(state1 + state2), 1e-3);

  // Verify the peephole connection to the forgetgate (weight = 4) by
  // neutralizing its contibution and therefore dividing the LSTM state value
  // by 2.
  input.clear();
  input << -state4 << -(state1 + state2) * 4 << state4 << state4;
  hiddenLayer0.FeedForward(input, output);
  BOOST_REQUIRE_CLOSE(output(0), std::tanh((state1 + state2) / 2), 1e-3);

  // Verify the peephole connection to the inputgate (weight = 3) by
  // neutralizing its contibution and therefore dividing the provided input
  // by 2.
  input.clear();
  input << -(state1 + state2) / 2 * 3 << -state4 << std::atanh(state3)
        << state4;
  hiddenLayer0.FeedForward(input, output);
  BOOST_REQUIRE_CLOSE(output(0), std::tanh(state3 / 2), 1e-3);

  // Verify the peephole connection to the outputgate (weight = 5) by
  // neutralizing its contibution and therefore dividing the provided output
  // by 2.
  input.clear();
  input << -state4 << state4 << state4 << -state3 / 2 * 5;
  hiddenLayer0.FeedForward(input, output);
  BOOST_REQUIRE_CLOSE(output(0), std::tanh(state3 / 2) / 2, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();
