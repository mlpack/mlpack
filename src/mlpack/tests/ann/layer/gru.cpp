/**
 * @file gru.cpp
 * @author Zachary Ng
 *
 * Tests the GRU layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

void GenerateSineData(arma::cube& predictors,
                      arma::cube& responses,
                      size_t cols,
                      size_t slices)
{
  predictors.resize(1, cols, slices);
  responses.resize(1, cols, 1);
  arma::vec theta = arma::linspace(0, 100, 1000);

  for (size_t i = 0; i < cols; i++)
  {
    predictors.tube(0, i) = arma::sin(theta.rows(i, i + slices - 1));
    responses[i] = sin(theta[i + slices]);
  }
}

/**
 * Try to train a simple RNN using a GRU layer
 */
TEST_CASE("SineGRULayerTest", "[ANNLayerTest]")
{
  size_t rho = 50;
  RNN<MeanSquaredError> model(rho, true);
  model.Add<GRU>(20);
  model.Add<Linear>(1);

  arma::cube predictors, responses, inital, out;
  GenerateSineData(predictors, responses, 10, rho);

  // Distance from truth without training
  model.Predict(predictors, inital);
  arma::vec initalDiff = arma::vectorise(responses - inital);
  double initalDist = arma::dot(initalDiff, initalDiff);

  model.Train(predictors, responses);

  // Distance from truth after training
  model.Predict(predictors, out);
  arma::vec trainedDiff = arma::vectorise(responses - out);
  double trainedDist = arma::dot(trainedDiff, trainedDiff);

  // Make sure the training result is better
  REQUIRE(initalDist - 1.0 > trainedDist);
}

/**
 * GRU layer numerical gradient test.
 */
TEST_CASE("GradientGRULayerTest", "[ANNLayerTest]")
{
  // GRU function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(4, 1, 1)),
        target(arma::ones(1, 1, 1))
    {
      model = RNN<MeanSquaredError, RandomInitialization>(5);
      model.ResetData(input, target);
      model.Add<GRU>(1);
      model.Add<Linear>(1);
      model.InputDimensions() = std::vector<size_t>{ 4 };
    }

    double Gradient(arma::mat& gradient)
    {
      return model.EvaluateWithGradient(model.Parameters(), 0, gradient, 1);
    }

    arma::mat& Parameters() { return model.Parameters(); }

    RNN<MeanSquaredError, RandomInitialization> model;
    arma::cube input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
