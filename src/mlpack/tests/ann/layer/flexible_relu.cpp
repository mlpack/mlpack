/**
 * @file tests/ann/layer/flexible_relu.cpp
 * @author Aarush Gupta
 * @author Manthan-R-Sheth
 *
 * Tests the flexible relu layer modules.
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
 * Jacobian FlexibleReLU module test.
 */
TEST_CASE("JacobianFlexibleReLULayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    FlexibleReLU module;
    arma::mat moduleParams(module.WeightSize(), 1);
    module.CustomInitialize(moduleParams, module.WeightSize());
    module.SetWeights((double*) moduleParams.memptr());

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}

/**
 * Flexible ReLU layer numerical gradient test.
 */
TEST_CASE("GradientFlexibleReLULayerTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(2, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, RandomInitialization>(
          NegativeLogLikelihood(), RandomInitialization(0.1, 0.5));

      model->ResetData(input, target);
      model->Add<Linear>(2);
      model->Add<LinearNoBias>(5);
      model->Add<FlexibleReLU>(0.05);
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

    FFN<NegativeLogLikelihood, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
