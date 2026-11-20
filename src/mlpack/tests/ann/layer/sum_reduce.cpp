/**
 * @file tests/ann/layer/sum_reduce.cpp
 * @author Andrew Furey
 *
 * Tests the ann SumReduce layer.
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
 * SumReduce input dimensions not set.
 */
TEST_CASE("SumReduceInputDimensionsNotSet", "[ANNLayerTest]")
{
  SumReduce layer(0);
  REQUIRE_THROWS_AS(layer.ComputeOutputDimensions(), std::logic_error);
}

/**
 * SumReduce axis is out of bounds.
 */
TEST_CASE("SumReduceAxisOutOfBounds", "[ANNLayerTest]")
{
  SumReduce layer(-1);
  layer.InputDimensions() = { 2, 2 };
  REQUIRE_THROWS_AS(layer.ComputeOutputDimensions(), std::logic_error);
}

/**
 * Set keep dimension to false.
 */
TEST_CASE("SumReduceKeepDimensionsFalse", "[ANNLayerTest]")
{
  SumReduce layer(1, false);
  layer.InputDimensions() = { 2, 8 };
  layer.ComputeOutputDimensions();

  std::vector<size_t> correctOutputDimensions = { 2 };
  REQUIRE(layer.OutputDimensions() == correctOutputDimensions);
}

/**
 * Set keep dimension to true.
 */
TEST_CASE("SumReduceKeepDimensionsTrue", "[ANNLayerTest]")
{
  SumReduce layer(1, true);
  layer.InputDimensions() = { 3, 2, 2 };
  layer.ComputeOutputDimensions();

  std::vector<size_t> correctOutputDimensions = { 3, 1, 2 };
  REQUIRE(layer.OutputDimensions() == correctOutputDimensions);
}

/**
 * Test sum reduce forward pass.
 */
TEST_CASE("SumReduceForwardPassAxis0", "[ANNLayerTest]")
{
  size_t axis = 0;

  size_t batchSize = 3;
  arma::mat input = arma::regspace(0, 8 * batchSize - 1);
  arma::mat output;

  arma::mat expectedOutput = arma::mat(
    { 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45 });
  expectedOutput.reshape(4, batchSize);

  SumReduce layer(axis, false);
  layer.InputDimensions() = { 2, 2, 2 };
  layer.ComputeOutputDimensions();

  input.reshape(8, batchSize);
  layer.Forward(input, output);
  CheckMatrices(output, expectedOutput);
}

/**
 * Test sum reduce forward pass.
 */
TEST_CASE("SumReduceForwardPassAxis1", "[ANNLayerTest]")
{
  size_t axis = 1;
  size_t batchSize = 3;
  arma::mat input = arma::regspace(0, 8 * batchSize - 1);
  arma::mat output;

  arma::mat expectedOutput = arma::mat(
    {2, 4, 10, 12, 18, 20, 26, 28, 34, 36, 42, 44});
  expectedOutput.reshape(4, batchSize);

  SumReduce layer(axis, false);
  layer.InputDimensions() = { 2, 2, 2 };
  layer.ComputeOutputDimensions();

  input.reshape(8, batchSize);
  layer.Forward(input, output);
  CheckMatrices(output, expectedOutput);
}

/**
 * Test sum reduce forward pass.
 */
TEST_CASE("SumReduceForwardPassAxis2", "[ANNLayerTest]")
{
  size_t axis = 2;
  size_t batchSize = 3;
  arma::mat input = arma::regspace(0, 12 * batchSize - 1);
  arma::mat output;

  arma::mat expectedOutput = arma::mat(
    {6, 8, 10, 12, 14, 16, 30, 32, 34, 36, 38, 40, 54, 56, 58, 60, 62, 64});
  expectedOutput.reshape(6, batchSize);

  SumReduce layer(axis, false);
  layer.InputDimensions() = { 3, 2, 2 };
  layer.ComputeOutputDimensions();

  input.reshape(12, batchSize);
  layer.Forward(input, output);
  CheckMatrices(output, expectedOutput);
}

/**
 * Test sum reduce backward pass.
 */
TEST_CASE("SumReduceBackwardPass", "[ANNLayerTest]")
{
  size_t batchSize = 3;
  arma::mat input, output;
  input.set_size(12, batchSize);

  arma::mat gy = arma::regspace(0, 6 * batchSize - 1);
  gy.reshape(6, batchSize);
  arma::mat g;

  SumReduce layer(2, false);
  layer.InputDimensions() = { 3, 2, 2 };
  layer.ComputeOutputDimensions();

  layer.Backward(input, output, gy, g);

  arma::mat expectedG = arma::mat(
    { 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
      6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17 });
  expectedG.reshape(12, batchSize);
  CheckMatrices(g, expectedG);
}

/**
 * JacobianTest for SumReduce layer
 */
TEST_CASE("SumReduceJacobianTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    arma::mat input(3 * 4 * 5, 1, arma::fill::randu);

    SumReduce layer(2, false);
    layer.InputDimensions() = { 3, 4, 5 };
    layer.ComputeOutputDimensions();

    double error = JacobianTest(layer, input);
    REQUIRE(error <= 1e-5);
  }
}
