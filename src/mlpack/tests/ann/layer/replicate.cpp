/**
 * @file tests/ann/layer/replicate.cpp
 * @author Adam Kropp
 *
 * Tests the replicate layer.
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
 * Simple test for Replicate layer.
 */
TEST_CASE("ReplicateTestCase1", "[ANNLayerTest]")
{
  // Input will be 4 x 3.
  arma::mat input = arma::mat(4, 3, arma::fill::randn);
  arma::mat target = arma::repmat(input, 2, 1);
  input.reshape(12, 1);
  target.reshape(24, 1);

  arma::mat output;
  // Output-Size should be 8 x 3.
  output.set_size(24, 1);


  Replicate module1({2, 1});
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 8);
  REQUIRE(module1.OutputDimensions()[1] == 3);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 24);
  REQUIRE(output.n_cols == 1);

  arma::mat prevDelta = arma::mat(24, 1, arma::fill::randn);
  arma::mat delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(8, 3);
  arma::mat targetDelta = (prevDelta.submat(0, 0, 3, 2) + prevDelta.submat(4, 0, 7, 2)) / 2;
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

/**
 * Simple test for Replicate layer.
 */
TEST_CASE("ReplicateTestCase2", "[ANNLayerTest]")
{
  // Input will be 4 x 3.
  arma::mat input = arma::mat(4, 3, arma::fill::randn);
  arma::mat target = arma::repmat(input, 1, 2);
  input.reshape(12, 1);
  target.reshape(24, 1);

  arma::mat output;
  // Output-Size should be 4 x 6.
  output.set_size(24, 1);


  Replicate module1({1, 2});
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 4);
  REQUIRE(module1.OutputDimensions()[1] == 6);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 24);
  REQUIRE(output.n_cols == 1);

  arma::mat prevDelta = arma::mat(24, 1, arma::fill::randn);
  arma::mat delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(4, 6);
  arma::mat targetDelta = (prevDelta.submat(0, 0, 3, 2)
      + prevDelta.submat(0, 3, 3, 5)) / 2;
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

/**
 * Simple test for Replicate layer.
 */
TEST_CASE("ReplicateTestCase3", "[ANNLayerTest]")
{
  // Input will be 4 x 3.
  arma::mat input = arma::mat(4, 3, arma::fill::randn);
  arma::mat target = arma::repmat(input, 2, 2);
  input.reshape(12, 1);
  target.reshape(48, 1);

  arma::mat output;
  // Output-Size should be 8 x 6.
  output.set_size(48, 1);


  Replicate module1({2, 2});
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 8);
  REQUIRE(module1.OutputDimensions()[1] == 6);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 48);
  REQUIRE(output.n_cols == 1);

  arma::mat prevDelta = arma::mat(48, 1, arma::fill::randn);
  arma::mat delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(8, 6);
  arma::mat targetDelta = (prevDelta.submat(0, 0, 3, 2)
      + prevDelta.submat(4, 0, 7, 2)
      + prevDelta.submat(0, 3, 3, 5)
      + prevDelta.submat(4, 3, 7, 5)) / 4;
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}
