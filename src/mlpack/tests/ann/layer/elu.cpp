/**
 * @file tests/ann/layer/elu.cpp
 * @author Ryan Curtin
 *
 * Tests the ELU layer.
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
 * ELU layer numerical gradient test.
 */
TEST_CASE("GradientELUTest", "[ANNLayerTest]")
{
  // Softmax function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1) - 0.5),
        target(arma::mat("1; 0"))
    {
      model = new FFN<MeanSquaredError, RandomInitialization>;
      model->ResetData(input, target);
      model->Add<Linear>(10);
      model->Add<ELU>(0.6);
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

/**
 * SELU layer numerical gradient test.
 */
TEST_CASE("GradientSELUTest", "[ANNLayerTest]")
{
  // Softmax function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1) - 0.5),
        target(arma::mat("1; 0"))
    {
      model = new FFN<MeanSquaredError, RandomInitialization>;
      model->ResetData(input, target);
      model->Add<Linear>(10);
      model->Add<SELU>();
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
