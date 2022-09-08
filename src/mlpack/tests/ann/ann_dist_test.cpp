/**
 * @file tests/ann_dist_test.cpp
 * @author Atharva Khandait
 * @author Nishant Kumar
 *
 * Tests the ann distributions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

/**
 * Simple bernoulli distribution module test.
 */
TEST_CASE("SimpleBernoulliDistributionTest", "[ANNDistTest]")
{
  arma::mat param = arma::mat("1 1 0");
  BernoulliDistribution<> module(param, false);

  arma::mat sample = module.Sample();
  // As the probabilities are [1, 1, 0], the bernoulli samples should be
  // [1, 1, 0] as well.
  CheckMatrices(param, sample);
}

/**
 * Jacobian bernoulli distribution module test when we don't apply logistic.
 */
TEST_CASE("JacobianBernoulliDistributionTest", "[ANNDistTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t targetElements = RandInt(2, 1000);

    arma::mat param;
    param.randn(targetElements, 1);

    arma::mat target;
    target.randn(targetElements, 1);

    BernoulliDistribution<> module(param, false);

    const double perturbation = 1e-6;
    double outputA, outputB, original;
    arma::mat jacobianA, jacobianB;

    // Initialize the jacobian matrix.
    jacobianA = arma::zeros(targetElements, 1);

    for (size_t j = 0; j < targetElements; ++j)
    {
      original = module.Probability()(j);
      module.Probability()(j) = original - perturbation;
      outputA = module.LogProbability(target);
      module.Probability()(j) = original + perturbation;
      outputB = module.LogProbability(target);
      module.Probability()(j) = original;
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j) = outputB;
    }

    module.LogProbBackward(target, jacobianB);
    REQUIRE(arma::max(arma::max(arma::abs(jacobianA - jacobianB))) <= 1e-5);
  }
}

/**
 * Jacobian bernoulli distribution module test when we apply logistic.
 */
TEST_CASE("JacobianBernoulliDistributionLogisticTest", "[ANNDistTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t targetElements = RandInt(2, 1000);

    arma::mat param;
    param.randn(targetElements, 1);

    arma::mat target;
    target.randn(targetElements, 1);

    BernoulliDistribution<> module(param);

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
      outputA = module.LogProbability(target);
      module.Logits()(j) = original + perturbation;
      LogisticFunction::Fn(module.Logits(), module.Probability());
      outputB = module.LogProbability(target);
      module.Logits()(j) = original;
      LogisticFunction::Fn(module.Logits(), module.Probability());
      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA(j) = outputB;
    }

    module.LogProbBackward(target, jacobianB);
    REQUIRE(arma::approx_equal(jacobianA, jacobianB, "both", 3e-5, 3e-5));
  }
}

/**
 * Normal Distribution module test.
 */
TEST_CASE("NormalDistributionTest", "[ANNDistTest]")
{
  arma::vec mu = {1.1, 1.2, 1.5, 1.7};
  arma::vec sigma = {0.1, 0.11, 0.5, 0.23};

  NormalDistribution<> normalDist(mu, sigma);

  arma::vec x = {1.05, 1.1, 1.7, 2.5};

  arma::vec prob;
  normalDist.LogProbability(x, prob);

  // Testing output of log probability for some random mu, sigma and x.
  REQUIRE(prob[0] == Approx(1.2586464).epsilon(1e-5));
  REQUIRE(prob[1] == Approx(0.8751131).epsilon(1e-5));
  REQUIRE(prob[2] == Approx(-0.30579138).epsilon(1e-5));
  REQUIRE(prob[3] == Approx(-5.498411).epsilon(1e-5));

  arma::vec dmu, dsigma;
  normalDist.ProbBackward(x, dmu, dsigma);

  // Testing output of dmu and dsigma for some random mu, sigma and x.
  REQUIRE(dmu[0] == Approx(-17.603287).epsilon(1e-5));
  REQUIRE(dsigma[0] == Approx(-26.40487).epsilon(1e-5));
  REQUIRE(dmu[1] == Approx(-19.827663).epsilon(1e-5));
  REQUIRE(dsigma[1] == Approx(-3.7852707).epsilon(1e-5));
  REQUIRE(dmu[2] == Approx(0.5892323).epsilon(1e-5));
  REQUIRE(dsigma[2] == Approx(-1.2373875).epsilon(1e-5));
  REQUIRE(dmu[3] == Approx(0.061901994).epsilon(1e-5));
  REQUIRE(dsigma[3] == Approx(0.19751444).epsilon(1e-5));
}

/**
 * Jacobian Normal Distribution module test for mean.
 */
TEST_CASE("JacobianNormalDistributionMeanTest", "[ANNDistTest]")
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t targetElements = RandInt(2, 1000);

    arma::mat mu;
    mu.randn(targetElements, 1);

    arma::mat sigma;
    sigma.randu(targetElements, 1);

    arma::mat x;
    x.randn(targetElements, 1);

    NormalDistribution<> module(mu, sigma);

    const double perturbation = 1e-6;
    arma::mat output, outputA, outputB, jacobianA, jacobianB;

    // Initialize the jacobian matrix.
    module.Probability(x, output);
    jacobianA = arma::zeros(x.n_elem, output.n_elem);

    for (size_t j = 0; j < x.n_elem; ++j)
    {
      double original = module.Mean()(j);
      module.Mean()(j) = original - perturbation;
      module.Probability(x, outputA);
      module.Mean()(j) = original + perturbation;
      module.Probability(x, outputB);
      module.Mean()(j) = original;

      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA.row(j) = outputB.t();
    }

    // Initialize the derivative parameter.
    arma::mat deriv = arma::zeros(output.n_rows, output.n_cols);

    // Share the derivative parameter.
    arma::mat derivTemp = arma::mat(deriv.memptr(), deriv.n_rows, deriv.n_cols,
        false, false);

    // Initialize the jacobian matrix.
    jacobianB = arma::zeros(mu.n_elem, output.n_elem);

    for (size_t k = 0; k < derivTemp.n_elem; ++k)
    {
      deriv.zeros();
      derivTemp(k) = 1;

      arma::mat deltaMu, deltaSigma;
      module.ProbBackward(x, deltaMu, deltaSigma);

      jacobianB.col(k) = deltaMu % deriv;
    }

    REQUIRE(arma::approx_equal(jacobianA, jacobianB, "both", 5e-3, 5e-3));
  }
}

/**
 * Jacobian Normal Distribution module test for standard deviation.
 */
TEST_CASE("JacobianNormalDistributionStandardDeviationTest", "[ANNDistTest]")
{
  for (size_t i = 0; i < 5; i++)
  {
    const size_t targetElements = RandInt(2, 1000);

    arma::mat mu;
    mu.randn(targetElements, 1);

    arma::mat sigma;
    sigma.randu(targetElements, 1);

    arma::mat x;
    x.randn(targetElements, 1);

    NormalDistribution<> module(mu, sigma);

    const double perturbation = 1e-6;
    arma::mat output, outputA, outputB, jacobianA, jacobianB;

    // Initialize the jacobian matrix.
    module.Probability(x, output);
    jacobianA = arma::zeros(x.n_elem, output.n_elem);

    for (size_t j = 0; j < x.n_elem; ++j)
    {
      double original = module.StandardDeviation()(j);
      module.StandardDeviation()(j) = original - perturbation;
      module.Probability(x, outputA);
      module.StandardDeviation()(j) = original + perturbation;
      module.Probability(x, outputB);
      module.StandardDeviation()(j) = original;

      outputB -= outputA;
      outputB /= 2 * perturbation;
      jacobianA.row(j) = outputB.t();
    }

    // Initialize the derivative parameter.
    arma::mat deriv = arma::zeros(output.n_rows, output.n_cols);

    // Share the derivative parameter.
    arma::mat derivTemp = arma::mat(deriv.memptr(), deriv.n_rows, deriv.n_cols,
        false, false);

    // Initialize the jacobian matrix.
    jacobianB = arma::zeros(sigma.n_elem, output.n_elem);

    for (size_t k = 0; k < derivTemp.n_elem; ++k)
    {
      deriv.zeros();
      derivTemp(k) = 1;

      arma::mat deltaMu, deltaSigma;
      module.ProbBackward(x, deltaMu, deltaSigma);

      jacobianB.col(k) = deltaSigma % deriv;
    }

    REQUIRE(arma::approx_equal(jacobianA, jacobianB, "both", 5e-3, 5e-3));
  }
}
