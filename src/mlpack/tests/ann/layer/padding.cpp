/**
 * @file tests/ann/layer/padding.cpp
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
 * Simple padding layer test.
 */
TEST_CASE("SimplePaddingLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  Padding module(1, 2, 3, 4);
  module.InputDimensions() = std::vector<size_t>({ 2, 5 });
  module.ComputeOutputDimensions();

  // Test the Forward function.
  input = arma::randu(10, 1);
  size_t totalOutputDimensions = module.OutputDimensions()[0];
  for (size_t i = 1; i < module.OutputDimensions().size(); ++i)
    totalOutputDimensions *= module.OutputDimensions()[i];
  output.set_size(totalOutputDimensions, input.n_cols);
  output.randu();
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == Approx(arma::accu(output)));
  REQUIRE(output.n_rows == (5 * 12)); // 2x5 --> 5x12

  // Test the Backward function.
  delta.set_size(input.n_rows, input.n_cols);
  module.Backward(input, output, delta);
  CheckMatrices(delta, input);

  // Test forward function for multiple filters.
  // Here it's 3 filters with height = 224, width = 224
  // the output should be [226 * 226 * 3, 1] with 1 padding.
  module = Padding(1, 1, 1, 1);
  module.InputDimensions() = std::vector<size_t>({ 224, 224, 3 });
  module.ComputeOutputDimensions();

  input = arma::randu(224 * 224 * 3, 1);
  totalOutputDimensions = module.OutputDimensions()[0];
  for (size_t i = 1; i < module.OutputDimensions().size(); ++i)
    totalOutputDimensions *= module.OutputDimensions()[i];
  output.set_size(totalOutputDimensions, input.n_cols);
  output.randu();
  module.Forward(input, output);
  REQUIRE(arma::accu(input) == Approx(arma::accu(output)));
  REQUIRE(output.n_rows == (226 * 226 * 3));
  REQUIRE(output.n_cols == 1);

  // Test forward function for multiple batches with multiple filters.
  // Here it's 3 filters with height = 244, width = 244
  // the output should be [246 * 246 * 3, 3] with 1 padding.
  module.InputDimensions() = std::vector<size_t>({ 244, 244, 3 });
  module.ComputeOutputDimensions();
  totalOutputDimensions = module.OutputDimensions()[0];
  for (size_t i = 1; i < module.OutputDimensions().size(); ++i)
    totalOutputDimensions *= module.OutputDimensions()[i];

  input = arma::randu(244 * 244 * 3, 3);
  output.set_size(totalOutputDimensions, input.n_cols);
  output.randu();
  module.Forward(input, output);
  REQUIRE(output.n_rows == (246 * 246 * 3));
  REQUIRE(output.n_cols == 3);
  REQUIRE(arma::accu(input) == Approx(arma::accu(output)));
}
