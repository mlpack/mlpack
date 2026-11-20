/**
 * @file tests/ann/layer/softmin.cpp
 * @author Aditya Raj
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
using namespace std;

/**
 * Simple Softmin module test.
 */
TEST_CASE("SimpleSoftminLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output, gy, g;
  Softmin module;

  // Test the forward function.
  input = {{0.0, 0.1, 0.2},
           {1.0, 1.1, 1.2},
           {2.0, 2.1, 2.2},
           {2.9, 2.8, 2.5}};
  arma::mat actualOutput = {{0.641750, 0.636772, 0.623646},
                            {0.236086, 0.234255, 0.229426},
                            {0.086851, 0.086177, 0.084401},
                            {0.035311, 0.042794, 0.062526}};

  module.Forward(input, output);
  REQUIRE(accu(arma::abs(actualOutput - output)) ==
             Approx(0.0).margin(1e-4));

  // Test the backward function.
  gy = arma::zeros(input.n_rows, input.n_cols);
  gy(1) = 1;
  module.Backward(input, output, gy, g);
  arma::mat calculatedGradient = {{-0.1515, 0, 0},
                                  {0.1803, 0, 0},
                                  {-0.0205, 0, 0},
                                  {-0.0083, 0, 0}};

  REQUIRE(accu(arma::abs(calculatedGradient - g)) == Approx(0.0).margin(1e-04));
}
