/**
 * @file tests/ann/layer/linear3d.cpp
 * @author Marcus Edel
 * @author Praveen Ch
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
 * Simple Linear3D layer test.
 */
TEST_CASE("SimpleLinear3DLayerTest", "[ANNLayerTest]")
{
  const size_t inSize = 4;
  const size_t outSize = 1;
  const size_t nPoints = 2;
  const size_t batchSize = 1;
  arma::mat input, output, delta;

  // Create a Linear3D layer outside of a network, and then set its memory.
  Linear3D module(outSize);
  module.InputDimensions() = std::vector<size_t>({ 4, 2 });
  module.ComputeOutputDimensions();
  arma::mat weights(module.WeightSize(), 1);
  module.SetWeights(weights.memptr());

  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(inSize * nPoints, batchSize);
  output.set_size(outSize * nPoints, batchSize);
  module.Forward(input, output);
  REQUIRE(arma::accu(module.Bias())
      == Approx(arma::accu(output) / (nPoints * batchSize)).epsilon(1e-3));

  // Test the Backward function.
  delta.set_size(input.n_rows, input.n_cols);
  output.zeros();
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0);
}

/**
 * Jacobian Linear3D module test.
 */
TEST_CASE("JacobianLinear3DLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inSize = RandInt(2, 10);
    const size_t outSize = RandInt(2, 10);
    const size_t nPoints = RandInt(2, 10);
    const size_t batchSize = 1;

    arma::mat input;
    input.set_size(inSize * nPoints, batchSize);

    // Create a Linear3D layer outside a network and initialize its memory.
    Linear3D module(outSize);
    module.InputDimensions() = std::vector<size_t>({ inSize, nPoints });
    module.ComputeOutputDimensions();
    arma::mat weights(module.WeightSize(), 1);
    module.SetWeights(weights.memptr());

    module.Parameters().randu();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}

/**
 * Simple Gradient test for Linear3D layer.
 */
TEST_CASE("GradientLinear3DLayerTest", "[ANNLayerTest]")
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        inSize(4),
        outSize(1),
        nPoints(2),
        batchSize(4)
    {
      input = arma::randu(inSize * nPoints, batchSize);
      target = arma::zeros(outSize * nPoints, batchSize);
      target(0, 0) = 1;
      target(0, 3) = 1;
      target(1, 1) = 1;
      target(1, 2) = 1;

      model = new FFN<MeanSquaredError, RandomInitialization>();
      model->ResetData(input, target);
      model->Add<Linear3D>(outSize);
      model->InputDimensions() = std::vector<size_t>{ 4, 2 };
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<MeanSquaredError, RandomInitialization>* model;
    arma::mat input, target;
    const size_t inSize;
    const size_t outSize;
    const size_t nPoints;
    const size_t batchSize;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-7);
}
