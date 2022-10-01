/**
 * @file tests/ann_regularizer_test.cpp
 * @author Saksham Bansal
 *
 * Tests the ANN regularizer modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../catch.hpp"
#include "../serialization.hpp"
#include "ann_test_tools.hpp"

using namespace mlpack;

TEST_CASE("GradientL1RegularizerTest", "[ANNRegularizerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
      factor(0.6),
      reg(factor)
    {
      // Nothing to do here.
    }

    double Output(const arma::mat& weight, size_t i, size_t j)
    {
      return std::abs(weight(i, j)) * factor;
    }

    void Gradient(arma::mat& weight, arma::mat& gradient)
    {
      reg.Evaluate(weight, gradient);
    }

    double factor;
    L1Regularizer reg;
  } function;

  REQUIRE(CheckRegularizerGradient(function) <= 1e-4);
}

TEST_CASE("GradientL2RegularizerTest", "[ANNRegularizerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        factor(0.6),
        reg(factor)
    {
      // Nothing to do here.
    }

    double Output(const arma::mat& weight, size_t i, size_t j)
    {
      return weight(i, j) * weight(i, j) * factor;
    }

    void Gradient(arma::mat& weight, arma::mat& gradient)
    {
      reg.Evaluate(weight, gradient);
    }

    double factor;
    L2Regularizer reg;
  } function;

  REQUIRE(CheckRegularizerGradient(function) <= 1e-4);
}

TEST_CASE("GradientOrthogonalRegularizerTest", "[ANNRegularizerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        factor(0.6),
        reg(factor)
    {
      // Nothing to do here.
    }

    double Output(const arma::mat& weight, size_t /* i */, size_t /* j */)
    {
      arma::mat x = arma::abs(weight * weight.t() -
          arma::eye<arma::mat>(weight.n_rows, weight.n_cols)) * factor;
      return arma::accu(x);
    }

    void Gradient(arma::mat& weight, arma::mat& gradient)
    {
      reg.Evaluate(weight, gradient);
    }

    double factor;
    OrthogonalRegularizer reg;
  } function;

  REQUIRE(CheckRegularizerGradient(function) <= 1e-4);
}
