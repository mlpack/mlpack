/**
 * @file tests/ann/layer/c_relu.cpp
 * @author Marcus Edel
 * @author Dhawal Arora
 *
 * Tests the concatenated relu layer modules.
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
 * Basic test of the CReLU function.
 */
TEST_CASE("CReLUFunctionTest", "[ANNLayerTest]")
{
  const arma::colvec desiredActivations("0 3 0 6 24 \
                                         2 0 0 0 0");

  const arma::colvec desiredDerivatives("0 0 0 0 0");

  const arma::colvec activationData("-2.0 3.0 0.0 6.0 24.0");

  CReLU crelu;
  crelu.InputDimensions() = { activationData.n_elem };
  crelu.ComputeOutputDimensions();

  // Test the activation function using the entire vector as input.
  arma::colvec activations(2 * activationData.n_elem);
  crelu.Forward(activationData, activations);
  arma::colvec derivatives(activationData.n_elem);
  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(desiredActivations.n_elem);
  crelu.Backward(desiredActivations, error, derivatives);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) ==
        Approx(desiredActivations.at(i)).epsilon(1e-5));
  }
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) ==
        Approx(desiredDerivatives.at(i)).epsilon(1e-5));
  }
}
