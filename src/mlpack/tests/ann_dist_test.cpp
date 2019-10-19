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

#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNDistTest);

/**
 * Simple bernoulli distribution module test.
 */
BOOST_AUTO_TEST_CASE(SimpleBernoulliDistributionTest)
{
  arma::mat param = arma::mat("1 1 0");
  BernoulliDistribution<> module(std::move(param), false);

  arma::mat sample = module.Sample();
  // As the probabilities are [1, 1, 0], the bernoulli samples should be
  // [1, 1, 0] as well.
  CheckMatrices(param, sample);
}

/**
 * Jacobian bernoulli distribution module test when we don't apply logistic.
 */
BOOST_AUTO_TEST_CASE(JacobianBernoulliDistributionTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t targetElements = math::RandInt(2, 1000);

    arma::mat param;
    param.randn(targetElements, 1);

    arma::mat target;
    target.randn(targetElements, 1);

    BernoulliDistribution<> module(std::move(param), false);

    const double perturbation = 1e-6;
    double outputA, outputB, original;
    arma::mat jacobianA, jacobianB;

    // Initialize the jacobian matrix.
    jacobianA = arma::zeros(targetElements, 1);

    for (size_t j = 0; j < targetElements; ++j)
    {
      original = module.Probability()(j);
      module.Probability()(j) = original - perturbation;
      outputA = module.LogProbability(std::move(target));
      module.Probability()(j) = original + perturbation;
      outputB = module.LogProbability(std::move(target));
      module.Probability()(j) = original;
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j) = outputB;
    }

    module.LogProbBackward(std::move(target), std::move(jacobianB));
    BOOST_REQUIRE_LE(arma::max(arma::max(arma::abs(jacobianA - jacobianB))),
        1e-5);
  }
}

/**
 * Jacobian bernoulli distribution module test when we apply logistic.
 */
BOOST_AUTO_TEST_CASE(JacobianBernoulliDistributionLogisticTest)
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t targetElements = math::RandInt(2, 1000);

    arma::mat param;
    param.randn(targetElements, 1);

    arma::mat target;
    target.randn(targetElements, 1);

    BernoulliDistribution<> module(std::move(param));

    const double perturbation = 1e-6;
    double outputA, outputB, original;
    arma::mat jacobianA, jacobianB;

    // Initialize the jacobian matrix.
    jacobianA = arma::zeros(targetElements, 1);

    for (size_t j = 0; j < targetElements; ++j)
    {
      original = module.Logits()(j);
      module.Logits()(j) = original - perturbation;
      LogisticFunction::Fn(module.Logits(), module.Probability());
      outputA = module.LogProbability(std::move(target));
      module.Logits()(j) = original + perturbation;
      LogisticFunction::Fn(module.Logits(), module.Probability());
      outputB = module.LogProbability(std::move(target));
      module.Logits()(j) = original;
      LogisticFunction::Fn(module.Logits(), module.Probability());
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j) = outputB;
    }

    module.LogProbBackward(std::move(target), std::move(jacobianB));
    BOOST_REQUIRE_LE(arma::max(arma::max(arma::abs(jacobianA - jacobianB))),
        3e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
