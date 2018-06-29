/**
 * @file ann_dist_test.cpp
 * @author Atharva Khandait
 *
 * Tests the ann distributions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/dists/normal_distribution.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNDistTest);

/**
 * Simple normal distribution module test.
 */
BOOST_AUTO_TEST_CASE(SimpleNormalDistributionTest)
{
  static const constexpr double log2pi = 1.83787706640934533908193770912475883;
  arma::mat param, target, gradient;
  double output;
  param.ones(10, 1);
  target.ones(5, 1);

  NormalDistribution<> module(std::move(param), false);

  // Test if the LogProbability function with the given input gives output
  // equal to manually evaluated output.
  output = module.LogProbability(std::move(target));
  BOOST_REQUIRE_EQUAL(output, -0.5 * log2pi * target.n_elem);

  // Test the Backward Log Probability function.
  module.LogProbBackward(std::move(target), std::move(gradient));
  BOOST_REQUIRE_EQUAL(arma::accu(gradient), -5);
}

/**
 * Jacobian normal distribution module test when we don't apply softplus.
 */
BOOST_AUTO_TEST_CASE(JacobianNormalDistributionTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t targetElements = math::RandInt(2, 1000);

    arma::mat param;
    param.randn(targetElements * 2, 1);

    arma::mat target;
    target.randn(targetElements, 1);

    NormalDistribution<> module(std::move(param), false);

    const double perturbation = 1e-6;
    double outputA, outputB, original;
    arma::mat jacobianA, jacobianB;

    // Initialize the jacobian matrix.
    jacobianA = arma::zeros(targetElements * 2, 1);

    for (size_t j = 0; j < targetElements; ++j)
    {
      original = module.StdDev()(j);
      module.StdDev()(j) = original - perturbation;
      outputA = module.LogProbability(std::move(target));
      module.StdDev()(j) = original + perturbation;
      outputB = module.LogProbability(std::move(target));
      module.StdDev()(j) = original;
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j) = outputB;

      original = module.Mean()(j);
      module.Mean()(j) = original - perturbation;
      outputA = module.LogProbability(std::move(target));
      module.Mean()(j) = original + perturbation;
      outputB = module.LogProbability(std::move(target));
      module.Mean()(j) = original;
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j + targetElements) = outputB;
    }

    module.LogProbBackward(std::move(target), std::move(jacobianB));
    BOOST_REQUIRE_LE(arma::max(arma::max(arma::abs(jacobianA - jacobianB))),
        1e-5);
  }
}

/**
 * Jacobian normal distribution module test when we apply softplus.
 */
BOOST_AUTO_TEST_CASE(JacobianNormalDistributionSoftplusTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t targetElements = math::RandInt(2, 1000);

    arma::mat param;
    param.randn(targetElements * 2, 1);

    arma::mat target;
    target.randn(targetElements, 1);

    NormalDistribution<> module(std::move(param));

    const double perturbation = 1e-6;
    double outputA, outputB, original;
    arma::mat jacobianA, jacobianB;

    // Initialize the jacobian matrix.
    jacobianA = arma::zeros(targetElements * 2, 1);

    for (size_t j = 0; j < targetElements; ++j)
    {
      original = module.PreStdDev()(j);
      module.PreStdDev()(j) = original - perturbation;
      module.ApplySoftplus();
      outputA = module.LogProbability(std::move(target));
      module.PreStdDev()(j) = original + perturbation;
      module.ApplySoftplus();
      outputB = module.LogProbability(std::move(target));
      module.PreStdDev()(j) = original;
      module.ApplySoftplus();
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j) = outputB;

      original = module.Mean()(j);
      module.Mean()(j) = original - perturbation;
      outputA = module.LogProbability(std::move(target));
      module.Mean()(j) = original + perturbation;
      outputB = module.LogProbability(std::move(target));
      module.Mean()(j) = original;
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j + targetElements) = outputB;
    }

    module.LogProbBackward(std::move(target), std::move(jacobianB));
    BOOST_REQUIRE_LE(arma::max(arma::max(arma::abs(jacobianA - jacobianB))),
        1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
