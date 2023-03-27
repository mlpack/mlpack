/**
 * @file tests/ann/layer/hard_tanh.cpp
 * @author Satyam Shukla
 *
 * Tests the parametric relu layer modules.
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
 * Hard TanH FORWARD Test.
 */
TEST_CASE("HardTanHFORWARDTest", "[ANNLayerTest]")
{
  arma::mat input = {{0.5, 1.2, 3.1},
                    {-2.2, -1.5, 0.8},
                    {5.5, -4.7, 2.1},
                    {0.2, 0.1, -0.5}};
  HardTanH module;
  arma::mat predOutput(input.n_rows,input.n_cols);
  module.Forward(input, predOutput);
  arma::mat actualOutput = {{0.5, 1.0, 1.0},
                           {-1.0, -1.0, 0.8},
                           {1, -1.0, 1.0},
                           {0.2, 0.1, -0.5}};
  REQUIRE(arma::accu(arma::abs(actualOutput - predOutput)) ==
      Approx(0.0).margin(1e-4));
}

/**
 * HardTanH BACKWARD Test.
 */
TEST_CASE("HardTanHBACKWARDTest", "[ANNLayerTest]")
{
  arma::mat input = {{0.5, 1.2, 3.1},
                    {-2.2, -1.5, 0.8},
                    {5.5, -4.7, 2.1},
                    {0.2, 0.1, -0.5}};
  HardTanH module;
  arma::mat gy = {{0.2, -0.5, 0.8},
                 {1.5, -0.6, 0.1},
                 {-0.3, 0.2, -0.5},
                 {0.1, -0.1, 0.3}};
  arma::mat predG(gy.n_rows,gy.n_cols);
  module.Backward(input, gy, predG);
  arma::mat actualG = {{0.2, 0.0, 0.0},
                      {0.0, 0.0, 0.1},
                      {0.0, 0.0, 0.0},
                      {0.1, -0.1, 0.3}};

  REQUIRE(arma::accu(arma::abs(actualG - predG)) ==
      Approx(0.0).margin(1e-4));
}
