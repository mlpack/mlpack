/**
 * @file tests/ann/layer/grouped_convolution.cpp
 * @author Shubham Agrawal
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
 * Simple test for the Grouped Convolution layer (With groups = 2).
 */
TEST_CASE("GroupedConvolutionLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output;

  // The input test matrix is of the form 3 x 2 x 4 x 1 where
  // number of images are 3 and number of feature maps are 2.
  input = { { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 } };

  GroupedConvolution layer(4, 1, 1, 2, 1, 1, 0, 0);
  layer.InputDimensions() = std::vector<size_t>({ 4, 1, 2 });
  layer.ComputeOutputDimensions();
  arma::mat layerWeights(layer.WeightSize(), 1);
  layerWeights(0) = -0.4886662960;
  layerWeights(1) = -0.4388893843;
  layerWeights(2) = -0.7269541025;
  layerWeights(3) = 0.7835500240;
  layerWeights(4) = -0.6865804195;
  layerWeights(5) = -0.7586858273;
  layerWeights(6) = -0.1721059084;
  layerWeights(7) = -0.1972532272;
  layer.SetWeights(layerWeights.memptr());
  output.set_size(layer.OutputSize(), 3);

  layer.Forward(input, output);

  // Value calculated using torch.nn.Conv2d().
  REQUIRE(arma::accu(output) == Approx(-647.6369628906).epsilon(1e-5));

  arma::mat delta;
  delta.set_size(8, 3);
  layer.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == Approx(686.7855224609).epsilon(1e-5));
}

/**
 * Test for testing equivalence of grouped convolution (groups = 1) 
 * with convolution layer.
 */
TEST_CASE("GroupedConvolutionEquivalenceTest", "[ANNLayerTest]")
{
  arma::mat input, output, outputG;

  // The input test matrix is of the form 3 x 2 x 2 x 2 where
  // number of images are 3 and number of feature maps are 2.
  input = { { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16 , 63 },
            { 32, 13 , 42 } };

  Convolution layer(2, 2, 2, 1, 1, 0, 0);
  layer.InputDimensions() = std::vector<size_t>({ 2, 2, 2 });
  layer.ComputeOutputDimensions();
  arma::mat layerWeights(layer.WeightSize(), 1);
  layerWeights(0) = 0.23757622;
  layerWeights(1) = -0.11899071;
  layerWeights(2) = 0.10450475;
  layerWeights(3) = -0.1303806;
  layerWeights(4) = -0.34706244;
  layerWeights(5) = -0.09472395;
  layerWeights(6) = 0.04117536;
  layerWeights(7) = -0.23012237;
  layerWeights(8) = -0.02827594;
  layerWeights(9) = -0.24280427;
  layerWeights(10) = 0.33375624;
  layerWeights(11) = -0.12285174;
  layerWeights(12) = -0.05546845;
  layerWeights(13) = -0.01502632;
  layerWeights(14) = -0.25894147;
  layerWeights(15) = -0.2283206;
  layerWeights(16) = 0.3204123974;
  layerWeights(17) = 0.2334779799;
  layer.SetWeights(layerWeights.memptr());
  output.set_size(layer.OutputSize(), 3);

  GroupedConvolution layerG(2, 2, 2, 1, 1, 1, 0, 0);
  layerG.InputDimensions() = std::vector<size_t>({ 2, 2, 2 });
  layerG.ComputeOutputDimensions();
  arma::mat layerWeightsG(layerG.WeightSize(), 1);
  layerWeightsG(0) = 0.23757622;
  layerWeightsG(1) = -0.11899071;
  layerWeightsG(2) = 0.10450475;
  layerWeightsG(3) = -0.1303806;
  layerWeightsG(4) = -0.34706244;
  layerWeightsG(5) = -0.09472395;
  layerWeightsG(6) = 0.04117536;
  layerWeightsG(7) = -0.23012237;
  layerWeightsG(8) = -0.02827594;
  layerWeightsG(9) = -0.24280427;
  layerWeightsG(10) = 0.33375624;
  layerWeightsG(11) = -0.12285174;
  layerWeightsG(12) = -0.05546845;
  layerWeightsG(13) = -0.01502632;
  layerWeightsG(14) = -0.25894147;
  layerWeightsG(15) = -0.2283206;
  layerWeightsG(16) = 0.3204123974;
  layerWeightsG(17) = 0.2334779799;
  layerG.SetWeights(layerWeightsG.memptr());
  outputG.set_size(layerG.OutputSize(), 3);

  layer.Forward(input, output);
  layerG.Forward(input, outputG);

  // Value calculated using torch.nn.Conv2d().
  CheckMatrices(output, outputG);
  REQUIRE(arma::accu(output) == Approx(12.6755657196).epsilon(1e-5));
  REQUIRE(arma::accu(outputG) == Approx(12.6755657196).epsilon(1e-5));

  arma::mat delta, deltaG;
  delta.set_size(8, 3);
  deltaG.set_size(8, 3);
  layer.Backward(input, output, delta);
  layerG.Backward(input, outputG, deltaG);

  CheckMatrices(delta, deltaG);
  REQUIRE(arma::accu(delta) == Approx(-1.9237523079).epsilon(1e-5));
  REQUIRE(arma::accu(deltaG) == Approx(-1.9237523079).epsilon(1e-5));
}

TEST_CASE("EdgeCaseFailGroupedConvolutionTest", "[ANNLayerTest]")
{
  GroupedConvolution layer(2, 2, 2, 0, 1, 1, 0, 0);
  layer.InputDimensions() = std::vector<size_t>({ 2, 2, 2 });
  REQUIRE_THROWS(layer.ComputeOutputDimensions());

  GroupedConvolution layer2(3, 2, 2, 2, 1, 1, 0, 0);
  layer2.InputDimensions() = std::vector<size_t>({ 2, 2, 2 });
  REQUIRE_THROWS(layer2.ComputeOutputDimensions());
}

/**
 * Grouped Convolution layer numerical gradient test.
 */
TEST_CASE("GradientGroupedConvolutionLayerTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randn(72, 2048)),
        target(arma::zeros(1, 2048))
    {
      model = new FFN<NegativeLogLikelihood, RandomInitialization>();
      model->ResetData(input, target);
      model->Add<GroupedConvolution>(4, 3, 3, 2, 1, 1, std::tuple<size_t, size_t>(0, 0),
          std::tuple<size_t, size_t>(0, 0), "same");
      model->Add<LogSoftMax>();

      model->InputDimensions() = std::vector<size_t>({ 6, 6, 2 });
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 2048);
      model->Gradient(model->Parameters(), 0, gradient, 2048);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) < 1e-1);
}
