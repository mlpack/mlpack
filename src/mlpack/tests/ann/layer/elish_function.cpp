/**
 * @file tests/ann/layer/flexible_relu.cpp
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
 * Simple test case for the Elish layer.
 */
TEST_CASE("ElishTest", "[ANNLayerTest]")
{

    // Create the Elish layer.
    ElishType<arma::colvec>* layer = new ElishType<arma::colvec>;

    // Input and output matrices.
    const arma::colvec input("-2 3.2 4.5 -100.2 1 -1 2 0");
    const arma::colvec desiredActivations("-0.10307056 3.0746696 4.4505587 \
                                         -3.0457406e-44 0.731058578 \
                                         -0.1700034 1.76159415 0.0 ");

    const arma::colvec desiredDerivatives("-0.074651888 1.0812559 1.0379111 \
                                          0 0.92767051 -0.025344425 1.0907842 \
                                          0.5 ");
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

TEST_CASE("JacobianElishTest", "[ANNLayerTest]")
{
    const size_t elems = arma::randi(arma::distr_param(2, 1000));

    arma::mat input(elems, 1);

    Elish module;
    module.InputDimensions() = { elems };
    module.ComputeOutputDimensions();
    module.Training() = true;

    double error = JacobianTest(module, input);
}