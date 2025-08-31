/**
 * @file tests/ann/layer/lookup.cpp
 * @author Kumar Utkarsh
 * @author 
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
 * Simple lookup module test.
 */
TEST_CASE("SimpleLookupLayerTest", "[ANNLayerTest]")
{
  const size_t vocabSize = 10;
  const size_t embeddingSize = 2;
  const size_t seqLength = 3;
  const size_t batchSize = 4;

  arma::mat output, input, gy, g, gradient;

  Lookup<> module(vocabSize, embeddingSize);
  arma::mat weights(embeddingSize, vocabSize); 
  module.InputDimensions() = std::vector<size_t>({ 3 });
  module.ComputeOutputDimensions();
  module.SetWeights(weights);

  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(seqLength, batchSize);
  module.Forward(input, output);
  REQUIRE(0 == accu(output));

  // No Need of Backward Test
}


// /**
//  * Lookup layer numerical gradient test.
//  */
// TEST_CASE("GradientLookupLayerTest", "[ANNLayerTest]")
// {
//   // Lookup function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction()
//     {
//       input.set_size(seqLength, batchSize);
//       for (size_t i = 0; i < input.n_elem; ++i)
//       {
//         input(i) = math::RandInt(1, vocabSize);
//       }
//       target = arma::zeros(vocabSize, batchSize);
//       for (size_t i = 0; i < batchSize; ++i)
//       {
//         const size_t targetWord = math::RandInt(1, vocabSize);
//         target(targetWord, i) = 1;
//       }

//       model = new FFN<CrossEntropyError<>, GlorotInitialization>();
//       model->Predictors() = input;
//       model->Responses() = target;
//       model->Add<Lookup<> >(vocabSize, embeddingSize);
//       model->Add<Linear<> >(embeddingSize * seqLength, vocabSize);
//       model->Add<Softmax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, batchSize);
//       model->Gradient(model->Parameters(), 0, gradient, batchSize);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<CrossEntropyError<>, GlorotInitialization>* model;
//     arma::mat input, target;

//     const size_t seqLength = 10;
//     const size_t embeddingSize = 8;
//     const size_t vocabSize = 20;
//     const size_t batchSize = 4;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-6);
// }
