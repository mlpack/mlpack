/**
 * @file tests/ann_layer_test.cpp
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
 * Simple test for AddMerge layer.
 */
TEST_CASE("AddMergeTestCase", "[ANNLayerTest]")
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

  AddMerge module1;
  module1.Add<MeanPooling>(2, 2, 2, 2, false);
  module1.Add<MeanPooling>(2, 2, 2, 2, false);

  AddMerge module2;
  module2.Add<MeanPooling>(2, 2, 2, 2, true);
  module2.Add<MeanPooling>(2, 2, 2, 2, true);

  module1.InputDimensions() = std::vector<size_t>({ 7, 4 });
  module1.ComputeOutputDimensions();
  module2.InputDimensions() = std::vector<size_t>({ 7, 4 });
  module2.ComputeOutputDimensions();

  // Calculated using torch.nn.MeanPool2d().
  arma::mat result1 = { { 1.5,  8.5 },
                        { 3.5,  8.0 },
                        { 5.5, 12.0 },
                        { 7.0,  5.0 } };

  arma::mat result2 = { { 1.5,  8.5 },
                        { 3.5,  8.0 }, 
                        { 5.5, 12.0 } };

  arma::mat output1, output2;
  output1.set_size(8, 1);
  output2.set_size(6, 1);
  module1.Forward(input, output1);
  REQUIRE(arma::accu(output1) == 51.0);
  module2.Forward(input, output2);
  REQUIRE(arma::accu(output2) == 39.0);
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
  REQUIRE(arma::accu(delta1) == Approx(21.6).epsilon(1e-3));
  module2.Backward(input, prevDelta2, delta2);
  REQUIRE(arma::accu(delta2) == Approx(16.2).epsilon(1e-3));
}

/**
 * Complex test for AddMerge layer.
 * This test includes: 
 * 1. AddMerge layer inside the AddMerge layer.
 * 2. Batch Size > 1.
 * 3. AddMerge layer with single child layer.
 */
TEST_CASE("AddMergeAdvanceTestCase", "[ANNLayerTest]")
{
  AddMerge r;
  AddMerge* r2 = new AddMerge();
  r2->Add<Linear>(5);
  r.Add<Linear>(5);
  r.Add(r2);
  r.InputDimensions() = std::vector<size_t>({ 5 });
  r.ComputeOutputDimensions();
  arma::mat rParams(r.WeightSize(), 1);
  r.SetWeights((double*) rParams.memptr());
  r.Network()[0]->Parameters().fill(2.0);
  ((AddMerge*) r.Network()[1])->Network()[0]->Parameters().fill(-1.0);

  Linear l(5);
  l.InputDimensions() = std::vector<size_t>({ 5 });
  l.ComputeOutputDimensions();
  arma::mat lParams(l.WeightSize(), 1);
  l.SetWeights((double*) lParams.memptr());
  l.Parameters().fill(1.0);

  arma::mat input(arma::randn(5, 10));
  arma::mat output1, output2;
  output1.set_size(5, 10);
  output2.set_size(5, 10);

  r.Forward(input, output1);
  l.Forward(input, output2);

  CheckMatrices(output1, output2, 1e-3);

  arma::mat delta1, delta2;
  delta1.set_size(5, 10);
  delta2.set_size(5, 10);
  r.Backward(input, output1, delta1);
  l.Backward(input, output2, delta2);

  CheckMatrices(output1, output2, 1e-3);
}
