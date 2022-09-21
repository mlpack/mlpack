/**
 * @file tests/ann/layer/linear_no_bias.cpp
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
 * Simple linear no bias module test.
 */
TEST_CASE("SimpleLinearNoBiasLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  LinearNoBias module(10);
  arma::mat weights(10 * 10, 1);
  module.InputDimensions() = std::vector<size_t>({ 10 });
  module.ComputeOutputDimensions();
  module.SetWeights(weights.memptr());

  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  REQUIRE(0 == arma::accu(output));

  // Test the Backward function.
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0);
}

/**
 * Jacobian linear no bias module test.
 */
TEST_CASE("JacobianLinearNoBiasLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = RandInt(2, 1000);
    const size_t outputElements = RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    LinearNoBias module(outputElements);
    arma::mat weights(inputElements * outputElements, 1);
    module.InputDimensions() = std::vector<size_t>({ inputElements });
    module.ComputeOutputDimensions();
    module.SetWeights(weights.memptr());

    module.Parameters().randu();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}

/**
 * LinearNoBias layer numerical gradient test.
 */
TEST_CASE("GradientLinearNoBiasLayerTest", "[ANNLayerTest]")
{
  // LinearNoBias function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(10);
      model->Add<LinearNoBias>(2);
      model->Add<LogSoftMax>();
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

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
