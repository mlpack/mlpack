/**
 * @file emfit_test.cpp
 * @author Kim SangYeon
 * 
 * Test for the EMFit class.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/gmm/em_fit.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::kmeans;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(EMFitTest);

//! Make sure the constructor of EMFit class works correctly.
BOOST_AUTO_TEST_CASE(EMFitConstructorTest)
{
  bool sameCluster = false;
  bool sameConstraint = false;

  // Build the fitter.
  EMFit<> fitter;

  // Check the default template parameters of EMFit class.
  // The EMFit class's default clusterer and constraint for covariance are
  // KMeans and PositiveDefiniteConstraint classes respectively.
  if (std::is_same<decltype(fitter.Clusterer()), KMeans<>&>::value)
    sameCluster = true;

  if (std::is_same<decltype(fitter.Constraint()),
    PositiveDefiniteConstraint&>::value)
    sameConstraint = true;

  // Check the default values are correct.
  // The EMFit class's default max-iteration is 300, and tolerance is 1e-10.
  BOOST_CHECK_EQUAL(fitter.MaxIterations(), 300);
  BOOST_CHECK_CLOSE(fitter.Tolerance(), 1e-10, 1e-3);
  BOOST_CHECK_EQUAL(sameCluster, true);
  BOOST_CHECK_EQUAL(sameConstraint, true);
}

/**
 * Make sure EMFit can estimate parameters reasonably given observations
 * from the Gaussian distributions.
 */
BOOST_AUTO_TEST_CASE(EMFitEstimateTest)
{
  // Create a list of Gaussian distributions.
  std::vector<GaussianDistribution> dists;
  GaussianDistribution d1("3.0 2.8", "1.2 0.2;"
                                     "0.2 0.8");

  GaussianDistribution d2("-5.0 -4.5", "2.2 0.7;"
                                       "0.7 1.2");

  GaussianDistribution d3("6.0 -5.7", "1.3 0.0;"
                                      "0.0 1.5");

  dists.push_back(d1);
  dists.push_back(d2);
  dists.push_back(d3);

  // Set the weights.
  arma::vec weights("0.2 0.3 0.5");

  // Generate the observation according to weights of the each distribution.
  arma::mat observations(2, 10000);
  for (size_t i = 0; i < 10000; i++)
  {
    double randValue = math::Random();

    if (randValue < weights[0]) // p(d1) = 0.2
      observations.col(i) = d1.Random();
    else if (randValue < (weights[0] + weights[1])) // p(d2) = 0.3
      observations.col(i) = d2.Random();
    else // p(d3) = 0.5
      observations.col(i) = d3.Random();
  }

  // Build the fitter.
  EMFit<> fitter;

  // Estimate the parameters.
  fitter.Estimate(observations, dists, weights, false);

  // Sort by the estimated weights for comparison.
  arma::uvec sortedIndices = arma::sort_index(weights);

  // Check the weights.
  BOOST_REQUIRE_SMALL(weights[sortedIndices[0]] - 0.2, 0.1);
  BOOST_REQUIRE_SMALL(weights[sortedIndices[1]] - 0.3, 0.1);
  BOOST_REQUIRE_SMALL(weights[sortedIndices[2]] - 0.5, 0.1);

  // Check the means and covariances.
  // First Gaussian (d1).
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_SMALL(dists[sortedIndices[0]].Mean()[i] -
        d1.Mean()[i], 0.2);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(dists[sortedIndices[0]].Covariance()(i, j) -
          d1.Covariance()(i, j), 0.25);

  // Second Gaussian (d2).
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_SMALL(dists[sortedIndices[1]].Mean()[i] -
        d2.Mean()[i], 0.2);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(dists[sortedIndices[1]].Covariance()(i, j) -
          d2.Covariance()(i, j), 0.25);

  // Third Gaussian (d3).
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_SMALL(dists[sortedIndices[2]].Mean()[i] -
        d3.Mean()[i], 0.2);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(dists[sortedIndices[2]].Covariance()(i, j) -
          d3.Covariance()(i, j), 0.25);
}

/**
 * Make sure EMFit can estimate parameters reasonably given observations
 * from the Gaussian distributions with probabilities of each point being from
 * the model.
 */
BOOST_AUTO_TEST_CASE(EMFitEstimateWithProbabilitiesTest)
{
  // Create a list of Gaussian distributions.
  std::vector<GaussianDistribution> dists;
  GaussianDistribution d1("3.0 2.8", "1.2 0.2;"
                                     "0.2 0.8");

  GaussianDistribution d2("-5.0 -4.5", "2.2 0.7;"
                                       "0.7 1.2");

  GaussianDistribution d3("6.0 -5.7", "1.3 0.0;"
                                      "0.0 1.5");
  dists.push_back(d1);
  dists.push_back(d2);
  dists.push_back(d3);

  // Set the weights.
  arma::vec weights("0.2 0.3 0.5");

  // Generate the observation according to weights of the each distribution.
  arma::mat observations(2, 10000);
  for (size_t i = 0; i < 10000; i++)
  {
    double randValue = math::Random();
    if (randValue < weights[0]) // p(d1) = 0.2
      observations.col(i) = d1.Random();
    else if (randValue < (weights[0] + weights[1])) // p(d2) = 0.3
      observations.col(i) = d2.Random();
    else // p(d3) = 0.5
      observations.col(i) = d3.Random();
  }

  arma::vec probabilities(10000);
  probabilities.randu();

  // Build the fitter.
  EMFit<> fitter;

  // Estimate the parameters.
  fitter.Estimate(observations, probabilities, dists, weights, false);

  // Sort by the estimated weights for comparison.
  arma::uvec sortedIndices = arma::sort_index(weights);

  // Check the weights.
  BOOST_REQUIRE_SMALL(weights[sortedIndices[0]] - 0.2, 0.1);
  BOOST_REQUIRE_SMALL(weights[sortedIndices[1]] - 0.3, 0.1);
  BOOST_REQUIRE_SMALL(weights[sortedIndices[2]] - 0.5, 0.1);

  // Check the means and covariances of the each gaussian.
  // First Gaussian (d1).
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_SMALL(dists[sortedIndices[0]].Mean()[i] -
        d1.Mean()[i], 0.2);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(dists[sortedIndices[0]].Covariance()(i, j) -
          d1.Covariance()(i, j), 0.31);

  // Second Gaussian (d2).
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_SMALL(dists[sortedIndices[1]].Mean()[i] -
        d2.Mean()[i], 0.2);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(dists[sortedIndices[1]].Covariance()(i, j) -
          d2.Covariance()(i, j), 0.31);

  // Third Gaussian (d3).
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_SMALL(dists[sortedIndices[2]].Mean()[i] -
        d3.Mean()[i], 0.2);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(dists[sortedIndices[2]].Covariance()(i, j) -
          d3.Covariance()(i, j), 0.31);
}

BOOST_AUTO_TEST_SUITE_END();
