/**
 * @file tests/ann/layer/fractional_max_pooling2d.cpp
 * @author Mayank Raj
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


TEST_CASE("FractionalMaxPooling2DTestCase", "[ANNLayerTest]")
{
  arma::mat input, output, delta;

  // Test case 1: Simple 4x4 input with a pooling ratio of 2.
  input = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12},
    {13, 14, 15, 16}
  };

  FractionalMaxPooling2DType<arma::mat> module1(2.0);
  module1.Forward(input, output);

  REQUIRE(output(0, 0) == 6);
  REQUIRE(output(0, 1) == 8);
  REQUIRE(output(1, 0) == 14);
  REQUIRE(output(1, 1) == 16);
  
  delta.set_size(input.n_rows, input.n_cols);
  module1.Backward(input, output, delta);

  double poolingArea = 2.0 * 2.0;
  REQUIRE(arma::accu(delta.submat(0, 0, 1, 1)) == 6.0 / poolingArea);
  REQUIRE(arma::accu(delta.submat(0, 2, 1, 3)) == 8.0 / poolingArea);
  REQUIRE(arma::accu(delta.submat(2, 0, 3, 1)) == 14.0 / poolingArea);
  REQUIRE(arma::accu(delta.submat(2, 2, 3, 3)) == 16.0 / poolingArea);

  // Test case 2: Different pooling ratio and input size.
  input = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };

  FractionalMaxPooling2DType<arma::mat> module2(1.5);
  module2.Forward(input, output);

  REQUIRE(output(0, 0) == 5);
  REQUIRE(output(0, 1) == 6);
  REQUIRE(output(1, 0) == 8);
  REQUIRE(output(1, 1) == 9);

  delta.set_size(input.n_rows, input.n_cols);
  module2.Backward(input, output, delta);

  poolingArea = 1.5 * 1.5;
  REQUIRE(arma::accu(delta.submat(0, 0, 1, 1)) == 5.0 / poolingArea);
  REQUIRE(arma::accu(delta.submat(0, 2, 1, 2)) == 6.0 / poolingArea);
  REQUIRE(arma::accu(delta.submat(2, 0, 2, 1)) == 8.0 / poolingArea);
  REQUIRE(arma::accu(delta.submat(2, 2, 2, 2)) == 9.0 / poolingArea);
}