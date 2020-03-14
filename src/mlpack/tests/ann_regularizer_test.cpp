/**
 * @file ann_regularizer_test.cpp
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

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/regularizer/regularizer.hpp>

#include <boost/test/unit_test.hpp>
#include "ann_test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNRegularizerTest);

BOOST_AUTO_TEST_CASE(GradientL1RegularizerTest)
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

  BOOST_REQUIRE_LE(CheckRegularizerGradient(function), 1e-4);
}

BOOST_AUTO_TEST_CASE(GradientL2RegularizerTest)
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

  BOOST_REQUIRE_LE(CheckRegularizerGradient(function), 1e-4);
}

BOOST_AUTO_TEST_CASE(GradientOrthogonalRegularizerTest)
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

  BOOST_REQUIRE_LE(CheckRegularizerGradient(function), 1e-4);
}

BOOST_AUTO_TEST_SUITE_END();
