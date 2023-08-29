/**
 * @file tests/ann/layer/elu.cpp
 * @author Satyam Shukla
 *
 * Tests the ELU layer.
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
 * Simple test case for the ELU layer.
 */
TEST_CASE("ELUTest", "[ANNLayerTest]")
{

    // Create the ELU layer.
    ELUType<arma::colvec>* layer = new ELUType<arma::colvec>;

    // Input and output matrices.
    const arma::colvec input("-2 3.2 4.5 -100.2 1 -1 2 0");
    const arma::colvec desiredActivations("-1.5201665 3.3622432 4.7281544 \
                                         -1.7580993 1.050701 -1.1113307 \
                                         2.101402 0");

    const arma::colvec desiredDerivatives("0.23793287 1.050701 1.050701 0 \
                                         1.050701 0.6467686 1.050701 \
                                         1.7580993 ");
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
    REQUIRE(arma::approx_equal(g, delta, "absdiff", 1e-5));
}

TEST_CASE("JacobianELUTest", "[ANNLayerTest]")
{
    const size_t elems = arma::randi(arma::distr_param(2, 1000));
    const int alpha = arma::randi<int>(arma::distr_param(1, 3));

    arma::mat input(elems, 1);

    ELU module(alpha);
    module.InputDimensions() = { elems };
    module.ComputeOutputDimensions();
    module.Training() = true;

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-3);
}