/**
 * @file tests/ann/layer/gelu.cpp
 * @author Satyam Shukla
 *
 * Tests the gelu layer.
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
 * Simple test case for the GELU layer.
 */
TEST_CASE("GELUTest", "[ANNLayerTest]")
{

    // Create the GELU layer.
    GELUType<arma::colvec>* layer = new GELUType<arma::colvec>;

    // Input and output matrices.
    const arma::colvec input("-2 3.2 4.5 -100.2 1 -1 2 0");
    const arma::colvec desiredActivations("-0.0454023 3.1981304 \
                                         4.5 -0.0 0.84119199 \
                                         -0.158808 1.954597694 0.0");

    const arma::colvec desiredDerivatives(" -0.086099 1.006493 1.000029 0.0 \
                                          1.082964  -0.082963 1.086099 0.5");
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

TEST_CASE("JacobianGELUTest", "[ANNLayerTest]")
{
    const size_t elems = arma::randi(arma::distr_param(2, 1000));

    arma::mat input(elems, 1);

    GELU module;
    module.InputDimensions() = { elems };
    module.ComputeOutputDimensions();
    module.Training() = true;

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-3);
}