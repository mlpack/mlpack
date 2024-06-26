/**
 * @file tests/layer/layer_norm.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 * @author Adam Kropp
 *
 * Tests the multihead_attention layer.
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
 * Tests the LayerNorm layer.
 */
TEST_CASE("LayerNormTest", "[ANNLayerTest]")
{
  arma::mat input, output;
  input = { { 5.1, 3.5 },
            { 4.9, 3.0 },
            { 4.7, 3.2 } };

  LayerNorm model;
  model.InputDimensions() = std::vector<size_t>({ 3 });
  model.ComputeOutputDimensions();
  arma::mat weights(model.WeightSize(), 1);
  model.SetWeights(weights);
  model.CustomInitialize(weights, model.WeightSize());

  model.Forward(input, output);

  arma::mat result;
  arma::mat mean = model.Mean();
  result = { 4.9000, 3.2333 };

  CheckMatrices(mean, result, 1e-1);
  result.clear();

  arma::mat var = model.Variance();
  result = { 0.0267, 0.0422 };

  CheckMatrices(var, result, 1e-1);

  result = { { 1.2247, 1.2978 },
             { 0, -1.1355 },
             { -1.2247, -0.1622 } };

  CheckMatrices(output, result, 1e-1);
  result.clear();
}

/**
 * LayerNorm layer numerical gradient test.
 */
TEST_CASE("GradientLayerNormTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randn(10, 256)),
        target(arma::zeros(1, 256))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Identity>();
      model->Add<Linear>(10);
      model->Add<LayerNorm>();
      model->Add<Linear>(2);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 16);
      model->Gradient(model->Parameters(), 0, gradient, 16);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

/**
 * Test that the functions that can access the parameters of the
 * Layer Norm layer work.
 */
TEST_CASE("LayerNormLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : size, eps.
  LayerNorm layer(1e-3);
  layer.InputDimensions() = std::vector<size_t>({ 5 });
  layer.ComputeOutputDimensions();

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.InSize() == 5);
  REQUIRE(layer.Epsilon() == 1e-3);
}
