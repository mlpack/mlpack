/**
 * @file tests/ann/layer/mean_pooling.cpp
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
 * Simple test for Mean Pooling layer.
 */
TEST_CASE("MeanPoolingTestCase", "[ANNLayerTest]")
{
  // For rectangular input to pooling layers.
  arma::mat input = arma::mat(28, 1);
  input.zeros();
  input(0) = input(16) = 1;
  input(1) = input(17) = 2;
  input(2) = input(18) = 3;
  input(3) = input(19) = 4;
  input(4) = input(20) = 5;
  input(5) = input(23) = 6;
  input(6) = input(24) = 7;
  input(14) = input(25) = 8;
  input(15) = input(26) = 9;

  MeanPooling module1(2, 2, 2, 2, false);
  MeanPooling module2(2, 2, 2, 2, true);
  module1.InputDimensions() = std::vector<size_t>({ 7, 4 });
  module1.ComputeOutputDimensions();
  module2.InputDimensions() = std::vector<size_t>({ 7, 4 });
  module2.ComputeOutputDimensions();

  // Calculated using torch.nn.MeanPool2d().
  arma::mat result1 = { { 0.75, 4.25 },
                        { 1.75, 4.00 },
                        { 2.75, 6.00 },
                        { 3.50, 2.50 } };

  arma::mat result2 = { { 0.75, 4.25 },
                        { 1.75, 4.00 },
                        { 2.75, 6.00 } };

  arma::mat output1, output2;
  output1.set_size(8, 1);
  output2.set_size(6, 1);
  module1.Forward(input, output1);
  REQUIRE(arma::accu(output1) == 25.5);
  module2.Forward(input, output2);
  REQUIRE(arma::accu(output2) == 19.5);
  output1.reshape(4, 2);
  output2.reshape(3, 2);
  CheckMatrices(output1, result1, 1e-1);
  CheckMatrices(output2, result2, 1e-1);

  arma::mat prevDelta1 = { { 3.6, -0.9 },
                           { 3.6, -0.9 },
                           { 3.6, -0.9 },
                           { 3.6, -0.9 } };

  arma::mat prevDelta2 = { { 3.6, -0.9 },
                           { 3.6, -0.9 },
                           { 3.6, -0.9 } };
  
  arma::mat delta1, delta2;
  delta1.set_size(28, 1);
  delta2.set_size(28, 1);
  prevDelta1.reshape(8, 1);
  prevDelta2.reshape(6, 1);
  module1.Backward(input, prevDelta1, delta1);
  REQUIRE(arma::accu(delta1) == Approx(10.8).epsilon(1e-3));
  module2.Backward(input, prevDelta2, delta2);
  REQUIRE(arma::accu(delta2) == Approx(8.1).epsilon(1e-3));
}
