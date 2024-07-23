/**
 * @file tests/ann/layer/concatenate.cpp
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
 * Simple concatenate module test.
 */
TEST_CASE("SimpleConcatenateLayerTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(5, 1);
  arma::mat output, delta;

  Concatenate module;
  module.Concat() = arma::ones(5, 1) * 0.5;
  module.InputDimensions() = std::vector<size_t>({ 5 });
  module.ComputeOutputDimensions();

  // Test the Forward function.
  output.set_size(module.OutputSize(), 1);
  module.Forward(input, output);

  REQUIRE(accu(output) == 7.5);

  // Test the Backward function.
  delta.set_size(5, 1);
  module.Backward(input, output, output, delta);
  REQUIRE(accu(delta) == 5);
}

/**
 * Concatenate layer numerical gradient test.
 */
TEST_CASE("GradientConcatenateLayerTest", "[ANNLayerTest]")
{
  // Concatenate function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(5);

      arma::mat concat = arma::ones(5, 1);
      // concatenate = new Concatenate();
      // concatenate->Concat() = concat;
      // model->Add(concatenate);
      model->Add<Concatenate>(concat);

      model->Add<Linear>(5);
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
    Concatenate* concatenate;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
