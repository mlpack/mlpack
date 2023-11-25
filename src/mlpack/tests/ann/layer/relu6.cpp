/**
 * @file tests/ann/layer/relu6.cpp
 * @author Marcus Edel
 * @author Dhawal Arora
 *
 * Tests the relu6 layer modules.
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
 * Implementation of the ReLU6 activation function derivative test. The function
 * is implemented as ReLU6 layer in the file relu6.hpp.
 *
 * @param input Input data used for evaluating the ReLU6 activation function.
 * @param target Target data used to evaluate the ReLU6 activation.
 */
void CheckReLU6Correct(const arma::colvec input,
                       const arma::colvec ActivationTarget,
                       const arma::colvec DerivativeTarget)
{
  // Initialize ReLU6 object.
  ReLU6 relu6;
  relu6.InputDimensions() = { input.n_rows };
  relu6.ComputeOutputDimensions();

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives, activations;
  activations.set_size(relu6.OutputDimensions()[0]);
  derivatives.set_size(input.n_rows);

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  relu6.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(ActivationTarget.at(i)).epsilon(1e-5));
  }
  relu6.Backward(input, activations, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(DerivativeTarget.at(i)).epsilon(1e-5));
  }
}

/**
 * Basic test of the ReLU6 function.
 */
TEST_CASE("ReLU6FunctionTest", "[ANNLayerTest]")
{
  const arma::colvec activationData("-2.0 3.0 0.0 6.0 24.0");

  // desiredActivations taken from PyTorch.
  const arma::colvec desiredActivations("0.0 3.0 0.0 6.0 6.0");

  // desiredDerivatives taken from PyTorch.
  const arma::colvec desiredDerivatives("0.0 1.0 0.0 0.0 0.0");

  CheckReLU6Correct(activationData, desiredActivations, desiredDerivatives);
}
