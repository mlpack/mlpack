/**
 * @file tests/ann/layer/lookup.cpp
 * @author Kumar Utkarsh
 *
 * Tests the Embedding layer.
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
 * Simple embedding module test.
 */
TEST_CASE("SimpleEmbeddingLayerTest", "[ANNLayerTest]")
{
  const size_t vocabSize = 10;
  const size_t embeddingSize = 10;
  const size_t seqLength = 3;
  const size_t batchSize = 4;

  arma::mat output, input;

  Embedding<> module(vocabSize, embeddingSize);
  arma::mat weights(embeddingSize, vocabSize);
  module.InputDimensions() = std::vector<size_t>({ 3 });
  module.ComputeOutputDimensions();
  module.SetWeights(weights);

  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(seqLength, batchSize);
  output.set_size(embeddingSize * seqLength, batchSize);
  module.Forward(input, output);
  for (size_t i = 0; i < batchSize; ++i)
  {
    REQUIRE(approx_equal(output.col(i), repmat(weights.col(0), seqLength, 1),
        "absdiff", 1e-5));
  }

  // No need for a backward pass test.
}

/**
 * Test that we can embed individual elements with manually crafted weights.
 */
TEST_CASE("ManualEmbeddingTest", "[ANNLayerTest]")
{
  const size_t vocabSize = 10;
  const size_t embeddingSize = 10;

  arma::mat input, output, weights;

  Embedding<> e(vocabSize, embeddingSize);
  weights.eye(vocabSize, embeddingSize);
  weights.diag() %= arma::linspace<arma::vec>(1, 10, 10);
  e.InputDimensions() = std::vector<size_t>({ 1 });
  e.ComputeOutputDimensions();
  REQUIRE(e.OutputDimensions().size() == 2);
  REQUIRE(e.OutputDimensions()[0] == embeddingSize);
  REQUIRE(e.OutputDimensions()[1] == 1);
  e.SetWeights(weights);

  input = arma::linspace<arma::rowvec>(0, 9, 10);
  output.set_size(embeddingSize, 10);
  e.Forward(input, output);

  REQUIRE(output.n_rows == embeddingSize);
  REQUIRE(output.n_cols == input.n_cols);
  for (size_t i = 0; i < 10; ++i)
    REQUIRE(approx_equal(output.col(i), weights.col(i), "absdiff", 1e-5));
}

/**
 * Embedding layer numerical gradient test.
 */
TEST_CASE("GradientEmbeddingLayerTest", "[ANNLayerTest]")
{
  // Embedding function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input.set_size(seqLength, batchSize);
      for (size_t i = 0; i < input.n_elem; ++i)
      {
        input(i) = RandInt(0, vocabSize - 1);
      }
      target = arma::zeros(vocabSize, batchSize);
      for (size_t i = 0; i < batchSize; ++i)
      {
        const size_t targetWord = RandInt(0, vocabSize - 1);
        target(targetWord, i) = 1;
      }

      model = FFN<CrossEntropyError, GlorotInitialization>();
      model.Add<Embedding>(vocabSize, embeddingSize);
      model.Add<Linear>(vocabSize);
      model.Add<Softmax>();
      model.InputDimensions() = std::vector<size_t>({ seqLength });
      model.ResetData(input, target);
    }

    double Gradient(arma::mat& gradient)
    {
      double error = model.Evaluate(model.Parameters(), 0, batchSize);
      model.Gradient(model.Parameters(), 0, gradient, batchSize);
      return error;
    }

    arma::mat& Parameters() { return model.Parameters(); }

    FFN<CrossEntropyError, GlorotInitialization> model;
    arma::mat input, target;

    const size_t seqLength = 10;
    const size_t embeddingSize = 8;
    const size_t vocabSize = 20;
    const size_t batchSize = 4;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-6);
}

/**
 * Test the embedding layer with attention directly after it.
 */
TEST_CASE("GradientEmbeddingMultiheadAttentionLayerTest", "[ANNLayerTest]")
{
  // Embedding function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input.set_size(seqLength, batchSize);
      for (size_t i = 0; i < input.n_elem; ++i)
      {
        input(i) = RandInt(0, vocabSize - 1);
      }
      target = arma::zeros(vocabSize, batchSize);
      for (size_t i = 0; i < batchSize; ++i)
      {
        const size_t targetWord = RandInt(0, vocabSize - 1);
        target(targetWord, i) = 1;
      }

      arma::cube attnMask = arma::zeros(seqLength, seqLength, batchSize);
      for (size_t k = 0; k < batchSize; ++k)
      {
        for (size_t i = 0; i < seqLength; ++i)
        {
          for (size_t j = 0; j < seqLength; ++j)
          {
            if (i < j)
              attnMask(i, j, k) = -std::numeric_limits<double>::infinity();
          }
        }
      }

      arma::mat keyPaddingMask = arma::zeros(seqLength, batchSize);
      keyPaddingMask.row(seqLength - 1).fill(
          -std::numeric_limits<double>::infinity());

      model = FFN<CrossEntropyError, GlorotInitialization>();
      model.Add<Embedding>(vocabSize, embeddingSize);
      model.Add<MultiheadAttention>(seqLength, numHeads, attnMask,
          keyPaddingMask, true);
      model.Add<Linear>(vocabSize);
      model.Add<Softmax>();
      model.InputDimensions() = std::vector<size_t>({ seqLength });
      model.ResetData(input, target);
    }

    double Gradient(arma::mat& gradient)
    {
      double error = model.Evaluate(model.Parameters(), 0, batchSize);
      model.Gradient(model.Parameters(), 0, gradient, batchSize);
      return error;
    }

    arma::mat& Parameters() { return model.Parameters(); }

    FFN<CrossEntropyError, GlorotInitialization> model;
    arma::mat input, target;

    const size_t seqLength = 10;
    const size_t embeddingSize = 8;
    const size_t vocabSize = 20;
    const size_t batchSize = 4;
    const size_t numHeads = 4;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-6);
}
