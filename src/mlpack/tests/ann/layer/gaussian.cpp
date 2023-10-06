/**
 * @file tests/ann/layer/gaussian.cpp
 * @author Satyam Shukla
 *
 * Tests the gaussian layer.
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
 * Simple test case for the Gaussian layer.
 */
TEST_CASE("GaussianTest", "[ANNLayerTest]")
{

    // Create the Gaussian layer.
    GaussianType<arma::colvec>* layer = new GaussianType<arma::colvec>;

    // Input and output matrices.
    const arma::colvec input("-2 3.2 0.5 -1.2 1 -1 2 0");
    const arma::colvec desiredActivations("0.018315 0.0000357 0.778800 \
                                        0.236927 0.367879 0.367879 0.018315 \
                                        1.0");

    const arma::colvec desiredDerivatives(" 0.073262 -0.000228 -0.778800 \
                                        0.568626 -0.735758 0.735758 -0.073262 \
                                        0.0");
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

TEST_CASE("JacobianGaussianTest", "[ANNLayerTest]")
{
    const size_t elems = arma::randi(arma::distr_param(2, 1000));

    arma::mat input(elems, 1);

    Gaussian module;
    module.InputDimensions() = { elems };
    module.ComputeOutputDimensions();
    module.Training() = true;

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-3);
}