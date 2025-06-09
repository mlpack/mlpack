/**
 * @file gru.cpp
 * @author Zachary Ng
 *
 * Tests the GRU layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

/**
 * GRU layer numerical gradient test.
 */
TEST_CASE("GradientGRULayerTest", "[ANNLayerTest]")
{
  // GRU function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(4, 1, 5)),
        target(arma::ones(1, 1, 5))
    {
      model = RNN<MeanSquaredError, RandomInitialization>(5);
      model.ResetData(input, target);
      model.Add<GRU>(1);
      model.Add<Linear>(1);
      model.InputDimensions() = std::vector<size_t>{ 4 };
    }

    double Gradient(arma::mat& gradient)
    {
      return model.EvaluateWithGradient(model.Parameters(), 0, gradient, 1);
    }

    arma::mat& Parameters() { return model.Parameters(); }

    RNN<MeanSquaredError, RandomInitialization> model;
    arma::cube input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
