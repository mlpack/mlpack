/**
 * @file tests/ann/layer/softmax.cpp
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
 * Simple Softmax module test.
 */
TEST_CASE("SimpleSoftmaxLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output, gy, g;
  Softmax module;

  // Test the forward function.
  input = arma::mat("1.7; 3.6");
  module.Forward(input, output);
  REQUIRE(arma::accu(arma::abs(arma::mat("0.130108; 0.869892") - output)) ==
      Approx(0.0).margin(1e-4));

  // Test the backward function.
  gy = arma::zeros(input.n_rows, input.n_cols);
  gy(0) = 1;
  module.Backward(output, gy, g);
  REQUIRE(arma::accu(arma::abs(arma::mat("0.11318; -0.11318") - g)) ==
      Approx(0.0).margin(1e-04));
}

/**
 * Softmax layer numerical gradient test.
 */
TEST_CASE("GradientSoftmaxTest", "[ANNLayerTest]")
{
  // Softmax function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("1; 0"))
    {
      model = new FFN<MeanSquaredError, RandomInitialization>;
      model->ResetData(input, target);
      model->Add<Linear>(10);
      model->Add<ReLU>();
      model->Add<Linear>(2);
      model->Add<Softmax>();
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

    FFN<MeanSquaredError>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
