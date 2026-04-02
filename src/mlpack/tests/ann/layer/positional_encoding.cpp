// /**
//  * @file tests/ann/layer/positional_encoding.cpp
//  * @author Kumar Utkarsh
//  *
//  * Tests the Positional Encoding layer.
//  *
//  * mlpack is free software; you may redistribute it and/or modify it under the
//  * terms of the 3-clause BSD license.  You should have received a copy of the
//  * 3-clause BSD license along with mlpack.  If not, see
//  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
//  */
// #include <mlpack/core.hpp>
// #include <mlpack/methods/ann.hpp>

// #include "../../test_catch_tools.hpp"
// #include "../../catch.hpp"
// #include "../../serialization.hpp"
// #include "../ann_test_tools.hpp"

// using namespace mlpack;

// /**
//  * Simple embedding module test.
//  */
// TEST_CASE("SimplePositionalEncodingLayerTest", "[ANNLayerTest]")
// {
//   const size_t embeddingSize = 10;
//   const size_t seqLength = 3;
//   const size_t batchSize = 4;

//   arma::mat output, input;

//   Embedding<> module(embeddingSize, seqLength);
//   module.InputDimensions() = std::vector<size_t>({ seqLength, embeddingSize });
//   module.ComputeOutputDimensions();

//   // Test the Forward function.
//   input = arma::zeros(embeddingSize * seqLength, batchSize);
//   output.set_size(embeddingSize * seqLength, batchSize);
//   module.Forward(input, output);
//   for (size_t i = 0; i < batchSize; ++i)
//   {
//     REQUIRE(approx_equal(output.col(i), repmat(weights.col(0), seqLength, 1),
//         "absdiff", 1e-5));
//   }

//   // No need for a backward pass test.
// }

// /**
//  * Test that we can embed individual elements with manually crafted weights.
//  */
// TEST_CASE("ManualEmbeddingTest", "[ANNLayerTest]")
// {
//   const size_t vocabSize = 10;
//   const size_t embeddingSize = 10;

//   arma::mat input, output, weights;

//   Embedding<> e(vocabSize, embeddingSize);
//   weights.eye(vocabSize, embeddingSize);
//   weights.diag() %= arma::linspace<arma::vec>(1, 10, 10);
//   e.InputDimensions() = std::vector<size_t>({ 1 });
//   e.ComputeOutputDimensions();
//   REQUIRE(e.OutputDimensions().size() == 2);
//   REQUIRE(e.OutputDimensions()[0] == embeddingSize);
//   REQUIRE(e.OutputDimensions()[1] == 1);
//   e.SetWeights(weights);

//   input = arma::linspace<arma::rowvec>(0, 9, 10);
//   output.set_size(embeddingSize, 10);
//   e.Forward(input, output);

//   REQUIRE(output.n_rows == embeddingSize);
//   REQUIRE(output.n_cols == input.n_cols);
//   for (size_t i = 0; i < 10; ++i)
//     REQUIRE(approx_equal(output.col(i), weights.col(i), "absdiff", 1e-5));
// }
