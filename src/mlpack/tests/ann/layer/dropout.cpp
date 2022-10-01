/**
 * @file tests/ann/layer/dropout.cpp
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
#include "../ann_test_tools.hpp"

using namespace mlpack;

/**
 * Simple dropout module test.
 */
TEST_CASE("SimpleDropoutLayerTest", "[ANNLayerTest]")
{
  // Initialize the probability of setting a value to zero.
  const double p = 0.2;

  // Initialize the input parameter.
  arma::mat input(1000, 1);
  input.fill(1 - p);

  Dropout module(p);
  module.Training() = true;

  // Test the Forward function.
  arma::mat output;
  module.Forward(input, output);
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(output) - (1 - p))) <= 0.05);

  // Test the Backward function.
  arma::mat delta;
  module.Backward(input, input, delta);
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(delta) - (1 - p))) <= 0.05);

  // Test the Forward function.
  module.Training() = false;
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == arma::accu(output));
}

/**
 * Perform dropout x times using ones as input, sum the number of ones and
 * validate that the layer is producing approximately the correct number of
 * ones.
 */
TEST_CASE("DropoutProbabilityTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  const size_t iterations = 10;

  double probability[5] = { 0.1, 0.3, 0.4, 0.7, 0.8 };
  for (size_t trial = 0; trial < 5; ++trial)
  {
    double nonzeroCount = 0;
    for (size_t i = 0; i < iterations; ++i)
    {
      Dropout module(probability[trial]);
      module.Training() = true;

      arma::mat output;
      module.Forward(input, output);

      // Return a column vector containing the indices of elements of X that
      // are non-zero, we just need the number of non-zero values.
      arma::uvec nonzero = arma::find(output);
      nonzeroCount += nonzero.n_elem;
    }
    const double expected = input.n_elem * (1 - probability[trial]) *
        iterations;
    const double error = fabs(nonzeroCount - expected) / expected;

    REQUIRE(error <= 0.15);
  }
}

/*
 * Perform dropout with probability 1 - p where p = 0, means no dropout.
 */
TEST_CASE("NoDropoutTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(1500, 1);
  Dropout module(0);
  module.Training() = true;

  arma::mat output;
  module.Forward(input, output);

  REQUIRE(arma::accu(output) == arma::accu(input));
}

