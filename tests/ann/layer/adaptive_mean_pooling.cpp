/**
 * @file tests/ann/layer/adaptive_mean_pooling.cpp
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
 * Simple test for Adaptive pooling for Mean Pooling layer.
 */
TEST_CASE("AdaptiveMeanPoolingTestCase", "[ANNLayerTest]")
{
  // For rectangular input.
  arma::mat input = arma::mat(12, 1);
  arma::mat output, delta;

  input.zeros();
  input(0) = 1;
  input(1) = 2;
  input(2) = 3;
  input(3) = input(8) = 7;
  input(4) = 4;
  input(5) = 5;
  input(6) = input(7) = 6;
  input(10) = 8;
  input(11) = 9;
  // Output-Size should be 2 x 2.
  // Square output.
  AdaptiveMeanPooling module1(2, 2);
  output.set_size(4, 1);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  module1.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  REQUIRE(accu(output) == 19.75);
  REQUIRE(output.n_elem == 4);
  REQUIRE(output.n_cols == 1);
  // Test the Backward Function.
  delta.set_size(12, 1);
  module1.Backward(input, output, output, delta);
  REQUIRE(accu(delta) == 19.75);

  // For Square input.
  input = arma::mat(9, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(3) = 3;
  input(6) = 3;
  // Output-Size should be 1 x 2.
  // Rectangular output.
  AdaptiveMeanPooling module2(1, 2);
  output.set_size(2, 1);
  module2.InputDimensions() = std::vector<size_t>({ 3, 3 });
  module2.ComputeOutputDimensions();
  module2.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  REQUIRE(accu(output) == 4.5);
  REQUIRE(output.n_elem == 2);
  REQUIRE(output.n_cols == 1);
  // Test the Backward Function.
  delta.set_size(9, 1);
  module2.Backward(input, output, output, delta);
  REQUIRE(accu(delta) == 4.50);

  // For Square input.
  input = arma::mat(16, 1);
  input.zeros();
  input(0) = 6;
  input(1) = 3;
  input(2) = 9;
  input(4) = 3;
  input(8) = 3;
  // Output-Size should be 3 x 3.
  // Square output.
  AdaptiveMeanPooling module3(3, 3);
  output.set_size(9, 1);
  module3.InputDimensions() = std::vector<size_t>({ 4, 4 });
  module3.ComputeOutputDimensions();
  module3.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  REQUIRE(accu(output) == 10.5);
  REQUIRE(output.n_elem == 9);
  REQUIRE(output.n_cols == 1);
  // Test the Backward Function.
  delta.set_size(16, 1);
  module3.Backward(input, output, output, delta);
  REQUIRE(accu(delta) == 10.5);

  // For Rectangular input.
  input = arma::mat(24, 1);
  input.zeros();
  input(0) = 3;
  input(1) = 3;
  input(4) = 3;
  // Output-Size should be 3 x 3.
  // Square output.
  AdaptiveMeanPooling module4(3, 3);
  output.set_size(9, 1);
  module4.InputDimensions() = std::vector<size_t>({ 6, 4 });
  module4.ComputeOutputDimensions();
  module4.Forward(input, output);
  // Calculated using torch.nn.AdaptiveAvgPool2d().
  REQUIRE(accu(output) == 2.25);
  REQUIRE(output.n_elem == 9);
  REQUIRE(output.n_cols == 1);
  // Test the Backward Function.
  delta.set_size(24, 1);
  module4.Backward(input, output, output, delta);
  REQUIRE(accu(delta) == 2.25);
}
