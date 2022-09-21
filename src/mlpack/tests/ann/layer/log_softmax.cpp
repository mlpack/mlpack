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
  input = arma::mat("0.5; 0.5");
  module.Forward(input, output);
  REQUIRE(arma::accu(arma::abs(arma::mat("-0.6931; -0.6931") - output)) ==
      Approx(0.0).margin(1e-3));

  // Test the Backward function.
  error = arma::zeros(input.n_rows, input.n_cols);
  // Assume LogSoftmax layer is always associated with NLL output layer.
  error(1, 0) = -1;
  module.Backward(input, error, delta);
  REQUIRE(arma::accu(arma::abs(arma::mat("1.6487; 0.6487") - delta)) ==
      Approx(0.0).margin(1e-3));
}
