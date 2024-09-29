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
  arma::mat moduleParams(module.WeightSize(), 1);
  module.CustomInitialize(moduleParams, module.WeightSize());
  module.SetWeights(moduleParams);
  arma::mat predOutput;
  module.Forward(input, predOutput);
  arma::mat actualOutput = {{0.5, 1.2, 3.1},
                           {-0.022, -0.015, 0.8},
                           {5.5, -0.047, 2.1},
                           {0.2, 0.1, -0.005}};
  REQUIRE(accu(arma::abs(actualOutput - predOutput)) ==
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
  module.SetWeights(moduleParams);
  arma::mat gy = {{0.2, -0.5, 0.8},
                 {1.5, -0.6, 0.1},
                 {-0.3, 0.2, -0.5},
                 {0.1, -0.1, 0.3}};
  arma::mat predG;
  arma::mat output;
  module.Forward(input, output);
  module.Backward(input, output, gy, predG);
  arma::mat actualG = {{0.2, -0.5, 0.8},
                      {0.015, -0.006, 0.1},
                      {-0.3, 0.002, -0.5},
                      {0.1, -0.1, 0.0030}};

  REQUIRE(accu(arma::abs(actualG - predG)) ==
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
  module.SetWeights(moduleParams);
  arma::mat error = {{0.2, -0.5,  0.8},
                    {-0.015, -0.006, 0.001},
                    {-0.3,  0.002, -0.005},
                    {0.1, -0.1, 0.0035}};
  arma::mat predGradient;
  module.Gradient(input, error, predGradient);

  REQUIRE(0.0103 - accu(predGradient) ==
      Approx(0.0).margin(1e-4));
}

double ComputeMSRE(arma::mat input, arma::mat target)
{
  return std::pow(accu(pow(input - target, 2)) / target.n_cols, 0.5);
}

TEST_CASE("PReLUIntegrationTest", "[ANNLayerTest]")
{
  arma::mat data;
  data::Load("boston_housing_price.csv", data);
  arma::mat labels;
  data::Load("boston_housing_price_responses.csv", labels);

  arma::mat trainData, testData, trainLabels, testLabels;
  data::Split(data, labels, trainData, testData, trainLabels, testLabels, 0.2);

  FFN<L1Loss> model;
  model.Add<Linear>(10);
  model.Add<PReLU>(0.01);
  model.Add<Linear>(3);
  model.Add<PReLU>(0.01);
  model.Add<Linear>(1);

  // Sometimes the model may not optimize correctly, so we allow a few trials.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    const size_t epochs = 250;
    ens::RMSProp optimizer(0.003, 8, 0.99, 1e-8, epochs * trainData.n_cols);
    model.Reset(data.n_rows);
    model.Train(trainData, trainLabels, optimizer);

    arma::mat predictions;
    model.Predict(trainData, predictions);
    double msreTrain = ComputeMSRE(predictions, trainLabels);
    model.Predict(testData, predictions);
    double msreTest = ComputeMSRE(predictions, testLabels);

    double relativeMSRE = std::abs((msreTest - msreTrain) / msreTrain);
    if (relativeMSRE <= 0.35)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

