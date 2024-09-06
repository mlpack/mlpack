/**
 * @file tests/ann/layer/log_softmax.cpp
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

/**
 * Simple LogSoftMax module test.
 */
TEST_CASE("SimpleLogSoftmaxLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, error, delta;
  LogSoftMax module;

  // Test the Forward function.
  input = arma::mat("-0.6871; 0.7898; 0.2011; 0.0949; -0.0550");
  module.Forward(input, output);
  REQUIRE(accu(arma::abs(
      arma::mat("-2.4746; -0.9977; -1.5864; -1.6926; -1.8425") - output)) ==
      Approx(0.0).margin(1e-3));

  // Test the Backward function.
  error = arma::ones(input.n_rows, input.n_cols);
  // Assume LogSoftmax layer is always associated with NLL output layer.
  module.Backward(input, output, error, delta);
  REQUIRE(accu(arma::abs(
      arma::mat("0.5790; -0.8435; -0.0233; 0.0798; 0.2079") - delta)) ==
      Approx(0.0).margin(1e-3));
}

/**
 * JacobianTest for LogSoftMax layer
 */
TEST_CASE("JacobianLogSoftMaxLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t elems = arma::randi(arma::distr_param(2, 1000));

    arma::mat input(elems, 1);

    LogSoftMax module;
    module.InputDimensions() = { elems };
    module.ComputeOutputDimensions();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}

