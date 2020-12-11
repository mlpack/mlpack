/**
 * @file tests/ann_visitor_test.cpp
 *
 * Tests for testing visitors in ANN's of mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/visitor/bias_set_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_set_visitor.hpp>
#include <mlpack/methods/ann/visitor/reset_visitor.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

/**
 * Test that the BiasSetVisitor works properly.
 */
TEST_CASE("BiasSetVisitorTest", "[ANNVisitorTest]")
{
  LayerTypes<> linear = new Linear<>(10, 10);

  arma::mat layerWeights(110, 1);
  layerWeights.zeros();

  ResetVisitor resetVisitor;

  boost::apply_visitor(WeightSetVisitor(layerWeights, 0), linear);

  boost::apply_visitor(resetVisitor, linear);

  arma::mat weight = {"1 2 3 4 5 6 7 8 9 10"};

  size_t biasSize = boost::apply_visitor(BiasSetVisitor(weight, 0), linear);

  REQUIRE(biasSize == 10);

  arma::mat input(10, 1), output;
  input.randu();

  boost::apply_visitor(ForwardVisitor(input, output), linear);

  REQUIRE(arma::accu(output) == 55);

  boost::apply_visitor(DeleteVisitor(), linear);
}

/**
 * Check correctness of WeightSize() for a layer.
 */
void CheckCorrectnessOfWeightSize(LayerTypes<>& layer)
{
  size_t weightSize = boost::apply_visitor(WeightSizeVisitor(),
      layer);

  arma::mat parameters;
  boost::apply_visitor(ParametersVisitor(parameters), layer);

  REQUIRE(weightSize == parameters.n_elem);
}

/**
 * Test that WeightSetVisitor works properly.
 */
TEST_CASE("WeightSetVisitorTest", "[ANNVisitorTest]")
{
  size_t randomSize = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> linear = new Linear<>(randomSize, randomSize);

  arma::mat layerWeights(randomSize * randomSize + randomSize, 1);
  layerWeights.zeros();

  size_t setWeights = boost::apply_visitor(WeightSetVisitor(layerWeights, 0),
      linear);

  REQUIRE(setWeights == randomSize * randomSize + randomSize);
}

/**
 * Test that WeightSizeVisitor works properly for linear layer.
 */
TEST_CASE("WeightSizeVisitorTestForLinearLayer", "[ANNVisitorTest]")
{
  size_t randomInSize = arma::randi(arma::distr_param(1, 100));
  size_t randomOutSize = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> linearLayer = new Linear<>(randomInSize, randomOutSize);

  CheckCorrectnessOfWeightSize(linearLayer);
}

/**
 * Test that WeightSizeVisitor works properly for concat layer.
 */
TEST_CASE("WeightSizeVisitorTestForConcatLayer", "[ANNVisitorTest]")
{
  LayerTypes<> concatLayer = new Concat<>();

  CheckCorrectnessOfWeightSize(concatLayer);
}

/**
 * Test that WeightSizeVisitor works properly for fast lstm layer.
 */
TEST_CASE("WeightSizeVisitorTestForFastLSTMLayer", "[ANNVisitorTest]")
{
  size_t randomInSize = arma::randi(arma::distr_param(1, 100));
  size_t randomOutSize = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> fastLSTMLayer = new FastLSTM<>(randomInSize, randomOutSize);

  CheckCorrectnessOfWeightSize(fastLSTMLayer);
}

/**
 * Test that WeightSizeVisitor works properly for Add layer.
 */
TEST_CASE("WeightSizeVisitorTestForAddLayer", "[ANNVisitorTest]")
{
  size_t randomOutSize = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> addLayer = new Add<>(randomOutSize);

  CheckCorrectnessOfWeightSize(addLayer);
}

/**
 * Test that WeightSizeVisitor works properly for Atrous Convolution Layer.
 */
TEST_CASE("WeightSizeVisitorTestForAtrousConvolutionLayer", "[ANNVisitorTest]")
{
  size_t randomInSize = arma::randi(arma::distr_param(1, 100));
  size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
  size_t randomKernelWidth = arma::randi(arma::distr_param(1, 100));
  size_t randomKernelHeight = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> atrousConvLayer = new AtrousConvolution<>(randomInSize,
      randomOutSize, randomKernelWidth, randomKernelHeight);

  CheckCorrectnessOfWeightSize(atrousConvLayer);
}


/**
 * Test that WeightSizeVisitor works properly for Convolution layer.
 */
TEST_CASE("WeightSizeVisitorTestForConvLayer", "[ANNVisitorTest]")
{
  size_t randomInSize = arma::randi(arma::distr_param(1, 100));
  size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
  size_t randomKernelWidth = arma::randi(arma::distr_param(1, 100));
  size_t randomKernelHeight = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> convLayer = new Convolution<>(randomInSize, randomOutSize,
      randomKernelWidth, randomKernelHeight);
  CheckCorrectnessOfWeightSize(convLayer);
}

/**
 * Test that WeightSizeVisitor works properly for BatchNorm layer.
 */
TEST_CASE("WeightSizeVisitorTestForBatchNormLayer", "[ANNVisitorTest]")
{
  size_t randomSize = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> batchNorm = new BatchNorm<>(randomSize);
  CheckCorrectnessOfWeightSize(batchNorm);
}

/**
 * Test that WeightSizeVisitor works properly for Transposed Convolution layer.
 */
TEST_CASE("WeightSizeVisitorTestForTransposedConvLayer", "[ANNVisitorTest]")
{
  size_t randomInSize = arma::randi(arma::distr_param(1, 100));
  size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
  size_t randomKernelWidth = arma::randi(arma::distr_param(1, 100));
  size_t randomKernelHeight = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> transposedConvLayer = new TransposedConvolution<>(randomInSize,
      randomOutSize, randomKernelWidth, randomKernelHeight);

  CheckCorrectnessOfWeightSize(transposedConvLayer);

  delete transposedConvLayer;
}

/**
 * Test that WeightSizeVisitor works properly for noisy linear layer.
 */
TEST_CASE("WeightSizeVisitorTestForNoisyLinearLayer", "[ANNVisitorTest]")
{
  size_t randomInSize = arma::randi(arma::distr_param(1, 100));
  size_t randomOutSize = arma::randi(arma::distr_param(1, 100));

  LayerTypes<> noisyLinearLayer = new NoisyLinear<>(randomInSize,
      randomOutSize);

  CheckCorrectnessOfWeightSize(noisyLinearLayer);

  delete noisyLinearLayer;
}
