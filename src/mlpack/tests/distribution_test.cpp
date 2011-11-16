/**
 * @file distribution_test.cpp
 * @author Ryan Curtin
 *
 * Test for the mlpack::distribution::DiscreteDistribution class.
 */
#include <mlpack/core.h>
#include <mlpack/methods/hmm/distributions/discrete_distribution.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(DistributionTest)

/**
 * Make sure we initialize correctly.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionConstructorTest)
{
  DiscreteDistribution d(5);

  BOOST_REQUIRE_EQUAL(d.Probabilities().n_elem, 5);
  BOOST_REQUIRE_CLOSE(d.Probability(0), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(1), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(2), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(3), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(4), 0.2, 1e-5);
}

/**
 * Make sure we get the probabilities of observations right.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionProbabilityTest)
{
  DiscreteDistribution d(5);

  d.Probabilities("0.2 0.4 0.1 0.1 0.2");

  BOOST_REQUIRE_CLOSE(d.Probability(0), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(1), 0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(2), 0.1, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(3), 0.1, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(4), 0.2, 1e-5);
}

/**
 * Make sure we get random observations correct.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionRandomTest)
{
  DiscreteDistribution d(3);

  d.Probabilities("0.3 0.6 0.1");

  arma::vec actualProb(3);

  for (size_t i = 0; i < 10000; i++)
    actualProb(d.Random())++;

  // Normalize.
  actualProb /= accu(actualProb);

  // 5% tolerance, because this can be a noisy process.
  BOOST_REQUIRE_CLOSE(actualProb(0), 0.3, 5.0);
  BOOST_REQUIRE_CLOSE(actualProb(1), 0.6, 5.0);
  BOOST_REQUIRE_CLOSE(actualProb(2), 0.1, 5.0);
}

/**
 * Make sure we can estimate from observations correctly.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionEstimateTest)
{
  DiscreteDistribution d(4);

  std::vector<size_t> obs;
  obs.push_back(0);
  obs.push_back(0);
  obs.push_back(1);
  obs.push_back(1);
  obs.push_back(2);
  obs.push_back(2);
  obs.push_back(2);
  obs.push_back(3);

  d.Estimate(obs);

  BOOST_REQUIRE_CLOSE(d.Probability(0), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(1), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(2), 0.375, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(3), 0.125, 1e-5);
}

/**
 * Estimate from observations with probabilities.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionEstimateProbTest)
{
  DiscreteDistribution d(3);

  std::vector<size_t> obs;
  obs.push_back(0);
  obs.push_back(0);
  obs.push_back(1);
  obs.push_back(2);

  std::vector<double> prob;
  prob.push_back(0.25);
  prob.push_back(0.25);
  prob.push_back(0.5);
  prob.push_back(1.0);

  d.Estimate(obs, prob);

  BOOST_REQUIRE_CLOSE(d.Probability(0), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(1), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability(2), 0.5, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
