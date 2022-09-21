/**
 * @file tests/ann/layer/alpha_dropout.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 *
 * Tests the ann layer modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

/*
 * Perform test to check whether mean and variance remain nearly same
 * after AlphaDropout.
 */
TEST_CASE("SimpleAlphaDropoutLayerTest", "[ANNLayerTest]")
{
  // Initialize the probability of setting a value to alphaDash.
  const double p = 0.2;

  // Initialize the input parameter having a mean nearabout 0
  // and variance nearabout 1.
  arma::mat input = arma::randn<arma::mat>(1000, 1);

  AlphaDropout module(p);
  module.Training() = true;

  // Test the Forward function when training phase.
  arma::mat output(arma::size(input));
  module.Forward(input, output);
  // Check whether mean remains nearly same.
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(input) - arma::mean(output))) <=
      0.15);

  // Check whether variance remains nearly same.
  REQUIRE(arma::as_scalar(arma::abs(arma::var(input) - arma::var(output))) <=
      0.15);

  // Test the Backward function when training phase.
  arma::mat delta;
  module.Backward(input, input, delta);
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(delta) - 0)) <= 0.1);

  // Test the Forward function when testing phase.
  module.Training() = false;
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == arma::accu(output));
}

/**
 * Perform AlphaDropout x times using ones as input, sum the number of ones
 * and validate that the layer is producing approximately the correct number
 * of ones.
 */
TEST_CASE("AlphaDropoutProbabilityTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      AlphaDropout module(probability[trial]);
      module.Training() = true;

      arma::mat output(arma::size(input));
      module.Forward(input, output);

      // Return a column vector containing the indices of elements of X
      // that are not alphaDash, we just need the number of
      // nonAlphaDash values.
      arma::uvec nonAlphaDash = arma::find(module.Mask());
      nonzeroCount += nonAlphaDash.n_elem;
    }

    const double expected = input.n_elem * (1-probability[trial]) * iterations;

    const double error = fabs(nonzeroCount - expected) / expected;

    REQUIRE(error <= 0.15);
  }
}

/**
 * Perform AlphaDropout with probability 1 - p where p = 0,
 * means no AlphaDropout.
 */
TEST_CASE("NoAlphaDropoutTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  AlphaDropout module(0);
  module.Training() = false;

  arma::mat output;
  module.Forward(input, output);

  REQUIRE(arma::accu(output) == arma::accu(input));
}

