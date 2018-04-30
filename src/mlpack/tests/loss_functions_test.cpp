/**
 * @file loss_functions_test.cpp
 * @author Dakshit Agrawal
 * @author Sourabh Varshney
 *
 * Tests for loss functions in mlpack::methods::ann:loss_functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/methods/ann/loss_functions/kl_divergence.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(LossFunctionsTest);

/**
 * Simple KL Divergence test.  The loss should be zero if input = target.
 */
BOOST_AUTO_TEST_CASE(SimpleKLDivergenceTest)
{
  arma::mat input, target, output;
  double loss;
  KLDivergence<> module(true);

  // Test the Forward function.  Loss should be 0 if input = target.
  input = arma::ones(10, 1);
  target = arma::ones(10, 1);
  loss = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_SMALL(loss, 0.00001);
}

/**
 * Test to check KL Divergence loss function when we take mean.
 */
BOOST_AUTO_TEST_CASE(KLDivergenceMeanTest)
{
  arma::mat input, target, output;
  double loss;
  KLDivergence<> module(true);

  // Test the Forward function.
  input = arma::mat("1 1 1 1 1 1 1 1 1 1");
  target = arma::exp(arma::mat("2 1 1 1 1 1 1 1 1 1"));

  loss = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_CLOSE_FRACTION(loss, -1.1 , 0.00001);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(target), std::move(output));
  BOOST_REQUIRE_CLOSE_FRACTION(arma::as_scalar(output), -0.1, 0.00001);
}

/**
 * Test to check KL Divergence loss function when we do not take mean.
 */
BOOST_AUTO_TEST_CASE(KLDivergenceNoMeanTest)
{
  arma::mat input, target, output;
  double loss;
  KLDivergence<> module(false);

  // Test the Forward function.
  input = arma::mat("1 1 1 1 1 1 1 1 1 1");
  target = arma::exp(arma::mat("2 1 1 1 1 1 1 1 1 1"));

  loss = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_CLOSE_FRACTION(loss, -11, 0.00001);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(target), std::move(output));
  BOOST_REQUIRE_CLOSE_FRACTION(arma::as_scalar(output), -1, 0.00001);
}

/*
 * Simple test for the mean squared error performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleMeanSquaredErrorLayerTest)
{
  arma::mat input, output, target;
  MeanSquaredError<> module;

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input = arma::mat("1.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0");
  target = arma::zeros(1, 8);
  double error = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_EQUAL(error, 0.5);

  // Test the Backward function.
  module.Backward(std::move(input), std::move(target), std::move(output));
  // We subtract a zero vector, so the output should be equal with the input.
  CheckMatrices(input, output);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the error function on a single input.
  input = arma::mat("2");
  target = arma::mat("3");
  error = module.Forward(std::move(input), std::move(target));
  BOOST_REQUIRE_EQUAL(error, 1.0);

  // Test the Backward function on a single input.
  module.Backward(std::move(input), std::move(target), std::move(output));
  // Test whether the output is negative.
  BOOST_REQUIRE_EQUAL(arma::accu(output), -1);
  BOOST_REQUIRE_EQUAL(output.n_elem, 1);
}

/*
 * Simple test for the cross-entropy error performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleCrossEntropyErrorLayerTest)
{
  arma::mat input1, input2, output, target1, target2;
  CrossEntropyError<> module(1e-6);

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module.Forward(std::move(input1), std::move(target1));
  BOOST_REQUIRE_SMALL(error1 - 8 * std::log(2), 2e-5);

  input2 = arma::mat("0 1 1 0 1 0 0 1");
  target2 = arma::mat("0 1 1 0 1 0 0 1");
  double error2 = module.Forward(std::move(input2), std::move(target2));
  BOOST_REQUIRE_SMALL(error2, 1e-5);

  // Test the Backward function.
  module.Backward(std::move(input1), std::move(target1), std::move(output));
  for (double el : output)
  {
    // For the 0.5 constant vector we should get 1 / (1 - 0.5) = 2 everywhere.
    BOOST_REQUIRE_SMALL(el - 2, 5e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input1.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input1.n_cols);

  module.Backward(std::move(input2), std::move(target2), std::move(output));
  for (size_t i = 0; i < 8; ++i)
  {
    double el = output.at(0, i);
    if (input2.at(i) == 0)
      BOOST_REQUIRE_SMALL(el - 1, 2e-6);
    else
      BOOST_REQUIRE_SMALL(el + 1, 2e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input2.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input2.n_cols);
}

/**
 * Simple test for the Sigmoid Cross Entropy Layer.
 */
BOOST_AUTO_TEST_CASE(SimpleSigmoidCrossEntropyLayerTest)
{
  arma::mat input1, input2, input3, output, target1,
            target2, target3, expectedOutput;
  SigmoidCrossEntropyError<> module;

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module.Forward(std::move(input1), std::move(target1));
  double expected = 0.97407699;
  // Value computed using tensorflow.
  BOOST_REQUIRE_SMALL(error1 / input1.n_elem - expected, 1e-7);

  input2 = arma::mat("1 2 3 4 5");
  target2 = arma::mat("0 0 1 0 1");
  double error2 = module.Forward(std::move(input2), std::move(target2));
  expected = 1.5027283;
  BOOST_REQUIRE_SMALL(error2 / input2.n_elem - expected, 1e-6);

  input3 = arma::mat("0 -1 -1 0 -1 0 0 -1");
  target3 = arma::mat("0 -1 -1 0 -1 0 0 -1");
  double error3 = module.Forward(std::move(input3), std::move(target3));
  expected = 0.00320443;
  BOOST_REQUIRE_SMALL(error3 / input3.n_elem - expected, 1e-6);

  // Test the Backward function.
  module.Backward(std::move(input1), std::move(target1), std::move(output));
  expected = 0.62245929;
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_SMALL(output(i) - expected, 1e-5);
  BOOST_REQUIRE_EQUAL(output.n_rows, input1.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input1.n_cols);

  expectedOutput = arma::mat(
      "0.7310586 0.88079709 -0.04742587 0.98201376 -0.00669285");
  module.Backward(std::move(input2), std::move(target2), std::move(output));
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_SMALL(output(i) - expectedOutput(i), 1e-5);
  BOOST_REQUIRE_EQUAL(output.n_rows, input2.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input2.n_cols);

  module.Backward(std::move(input3), std::move(target3), std::move(output));
  expectedOutput = arma::mat("0.5 1.2689414");
  for (size_t i = 0; i < 8; ++i)
  {
    double el = output.at(0, i);
    if (std::abs(input3.at(i) - 0.0) < 1e-5)
      BOOST_REQUIRE_SMALL(el - expectedOutput[0], 2e-6);
    else
      BOOST_REQUIRE_SMALL(el - expectedOutput[1], 2e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input3.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input3.n_cols);
}

BOOST_AUTO_TEST_SUITE_END();
