/**
 * @file tests/ann/layer/elish_function.cpp
 * @author Satyam Shukla
 *
 * Tests the elish layer.
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
 * Simple test case for the HardSwish layer.
 */
TEST_CASE("HardSwishTest", "[ANNLayerTest]")
{

    // Create the HardSwish layer.
    HardSwishType<arma::colvec>* layer = new HardSwishType<arma::colvec>;

    // Input and output matrices.
    const arma::colvec input("-2 3.2 4.5 -100.2 1 -1 2 0");
    const arma::colvec desiredActivations("-0.33333333 3.2 4.5 0 0.66666666 \
                                          -0.33333333 1.666666667 0.0 ");

    const arma::colvec desiredDerivatives("-0.16666666 1 1 0 0.833333334 \
                                         0.16666666 1.166666667 0.5 ");
    arma::colvec output;

    // Forward pass.
    layer -> Training() = true;
    layer -> Forward(input, output);

    // Test the Forward function.
    REQUIRE(arma::approx_equal(output, desiredActivations, "absdiff", 1e-5));


    arma::colvec gy(input.n_elem), g;
    gy.fill(1.0);


    // Backward pass.
    layer -> Backward(output, gy, g);

    // Test the Backward function.
    REQUIRE(arma::approx_equal(g, desiredDerivatives, "absdiff", 1e-5));
}

TEST_CASE("JacobianHardSwishTest", "[ANNLayerTest]")
{
    const size_t elems = arma::randi(arma::distr_param(2, 1000));

    arma::mat input(elems, 1);

    HardSwish module;
    module.InputDimensions() = { elems };
    module.ComputeOutputDimensions();
    module.Training() = true;

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-3);
}