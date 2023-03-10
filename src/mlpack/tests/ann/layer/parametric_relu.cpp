/**
 * @file tests/ann/layer/parametric_relu.cpp
 * @author Adarsh Santoria
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
 * PReLU FORWARD Test.
 */
TEST_CASE("PReLUFORWARDTest", "[ANNLayerTest]")
{
  arma::mat input = {{0.5, 1.2, 3.1},
                    {-2.2, -1.5, 0.8},
                    {5.5, -4.7, 2.1},
                    {0.2, 0.1, -0.5}};
  PReLU module(0.01);
  module.Training() = true;
  arma::mat moduleParams(module.WeightSize(), 1);
  module.CustomInitialize(moduleParams, module.WeightSize());
  module.SetWeights((double*) moduleParams.memptr());
  arma::mat predOutput;
  module.Forward(input, predOutput);
  arma::mat actualOutput = {{0.5, 1.2, 3.1},
                           {-0.022, -0.015, 0.8},
                           {5.5, -0.047, 2.1},
                           {0.2, 0.1, -0.005}};
  REQUIRE(arma::accu(arma::abs(actualOutput - predOutput)) ==
      Approx(0.0).margin(1e-4));
}

/**
 * PReLU BACKWARD Test.
 */
TEST_CASE("PReLUBACKWARDTest", "[ANNLayerTest]")
{
  arma::mat input = {{0.5, 1.2, 3.1},
                    {-2.2, -1.5, 0.8},
                    {5.5, -4.7, 2.1},
                    {0.2, 0.1, -0.5}};
  PReLU module(0.01);
  arma::mat moduleParams(module.WeightSize(), 1);
  module.CustomInitialize(moduleParams, module.WeightSize());
  module.SetWeights((double*) moduleParams.memptr());
  arma::mat gy = {{0.2, -0.5, 0.8},
                 {1.5, -0.6, 0.1},
                 {-0.3, 0.2, -0.5},
                 {0.1, -0.1, 0.3}};
  arma::mat predG;
  module.Backward(input, gy, predG);
  arma::mat actualG = {{0.2, -0.5, 0.8},
                      {0.015, -0.006, 0.1},
                      {-0.3, 0.002, -0.5},
                      {0.1, -0.1, 0.0030}};

  REQUIRE(arma::accu(arma::abs(actualG - predG)) ==
      Approx(0.0).margin(1e-4));
}

/**
 * PReLU GRADIENT Test.
 */
TEST_CASE("PReLUGRADIENTTest", "[ANNLayerTest]")
{
  arma::mat input = {{0.5, 1.2, 3.1},
                    {-2.2, -1.5, 0.8},
                    {5.5, -4.7, 2.1},
                    {0.2, 0.1, -0.5}};
  PReLU module(0.01);
  arma::mat moduleParams(module.WeightSize(), 1);
  module.CustomInitialize(moduleParams, module.WeightSize());
  module.SetWeights((double*) moduleParams.memptr());
  arma::mat error = {{0.2, -0.5,  0.8},
                    {-0.015, -0.006, 0.001},
                    {-0.3,  0.002, -0.005},
                    {0.1, -0.1, 0.0035}};
  arma::mat predGradient;
  module.Gradient(input, error, predGradient);

  REQUIRE(0.0103 - arma::accu(predGradient) ==
      Approx(0.0).margin(1e-4));
}
