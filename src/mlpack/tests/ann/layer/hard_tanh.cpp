/*
 * @file tests/ann/layer/hard_tanh.cpp
 * @author Vaibhav Pathak
 *
 * Tests the hard_tanh layer.
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
 * Simple HardTanH module test
 */

TEST_CASE("SimpleHardTanHTest", "[ANNLayerTest]")
{
  arma::mat output, gy, g;
  arma::mat input = {{-1.3743, -0.5565,  0.2742, -0.0151,  -1.4871},
    {1.5797, -4.2711, -2.2505, -1.7105, -1.2544},
    {0.4023,  0.5676, 2.3100, 1.6658, -0.1907},
    {0.1897,  0.9097, 0.1418, -1.5349, 0.1225},
    {-0.1101, -3.3656, -5.4033, -2.2240, -3.3235}};
  arma::mat actualOutput = {{-1.0000, -0.5565, 0.2742, -0.0151, -1.0000},
    {1.0000,  -1.0000,  -1.0000,  -1.0000,  -1.0000},
    {0.4023,   0.5676,   1.0000,   1.0000,  -0.1907},
    {0.1897,   0.9097,   0.1418,  -1.0000,   0.1225},
    {-0.1101,  -1.0000,  -1.0000,  -1.0000,  -1.0000}};

  HardTanH module;
	
  output.set_size(5,5);
  // Test the Forward function
  module.Forward(input, output);
  REQUIRE(arma::accu(output - actualOutput) == Approx(0).epsilon(1e-4));
  
  arma::mat delta = {{0  , 1.0, 1.0, 1.0, 0.0},
    {0  , 0  , 0  , 0.0, 0.0},
    {1.0, 1.0, 0  , 0.0, 1.0},
    {1.0, 1.0, 1.0, 0.0, 1.0},
    {1.0, 0  , 0.0, 0.0, 0.0}};

  gy.set_size(5,5);
  gy.fill(1);
  g.set_size(5,5);
  
  //Test the Backward function
  module.Backward(output, gy, g);
  REQUIRE(arma::accu(g - delta) == Approx(0).epsilon(1e-4));
}

