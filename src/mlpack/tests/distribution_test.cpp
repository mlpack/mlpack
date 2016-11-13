/**
 * @file distribution_test.cpp
 * @author Ryan Curtin
 * @author Yannis Mentekidis
 *
 * Tests for the classes:
 *  * mlpack::distribution::DiscreteDistribution
 *  * mlpack::distribution::GaussianDistribution
 *  * mlpack::distribution::GammaDistribution
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(DistributionTest);

/*********************************/
/** Discrete Distribution Tests **/
/*********************************/

/**
 * Make sure we initialize correctly.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionConstructorTest)
{
  DiscreteDistribution d(5);

  BOOST_REQUIRE_EQUAL(d.Probabilities().n_elem, 5);
  BOOST_REQUIRE_CLOSE(d.Probability("0"), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("1"), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("2"), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("3"), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("4"), 0.2, 1e-5);
}

/**
 * Make sure we get the probabilities of observations right.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionProbabilityTest)
{
  DiscreteDistribution d(5);

  d.Probabilities() = "0.2 0.4 0.1 0.1 0.2";

  BOOST_REQUIRE_CLOSE(d.Probability("0"), 0.2, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("1"), 0.4, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("2"), 0.1, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("3"), 0.1, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("4"), 0.2, 1e-5);
}

/**
 * Make sure we get random observations correct.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionRandomTest)
{
  DiscreteDistribution d(3);

  d.Probabilities() = "0.3 0.6 0.1";

  arma::vec actualProb(3);
  actualProb.zeros();

  for (size_t i = 0; i < 50000; i++)
    actualProb((size_t) (d.Random()[0] + 0.5))++;

  // Normalize.
  Log::Debug << actualProb.t();
  actualProb /= accu(actualProb);

  // 8% tolerance, because this can be a noisy process.
  BOOST_REQUIRE_CLOSE(actualProb(0), 0.3, 8.0);
  BOOST_REQUIRE_CLOSE(actualProb(1), 0.6, 8.0);
  BOOST_REQUIRE_CLOSE(actualProb(2), 0.1, 8.0);
}

/**
 * Make sure we can estimate from observations correctly.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionTrainTest)
{
  DiscreteDistribution d(4);

  arma::mat obs("0 0 1 1 2 2 2 3");

  d.Train(obs);

  BOOST_REQUIRE_CLOSE(d.Probability("0"), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("1"), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("2"), 0.375, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("3"), 0.125, 1e-5);
}

/**
 * Estimate from observations with probabilities.
 */
BOOST_AUTO_TEST_CASE(DiscreteDistributionTrainProbTest)
{
  DiscreteDistribution d(3);

  arma::mat obs("0 0 1 2");

  arma::vec prob("0.25 0.25 0.5 1.0");

  d.Train(obs, prob);

  BOOST_REQUIRE_CLOSE(d.Probability("0"), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("1"), 0.25, 1e-5);
  BOOST_REQUIRE_CLOSE(d.Probability("2"), 0.5, 1e-5);
}

/*********************************/
/** Gaussian Distribution Tests **/
/*********************************/

/**
 * Make sure Gaussian distributions are initialized correctly.
 */
BOOST_AUTO_TEST_CASE(GaussianDistributionEmptyConstructor)
{
  GaussianDistribution d;

  BOOST_REQUIRE_EQUAL(d.Mean().n_elem, 0);
  BOOST_REQUIRE_EQUAL(d.Covariance().n_elem, 0);
}

/**
 * Make sure Gaussian distributions are initialized to the correct
 * dimensionality.
 */
BOOST_AUTO_TEST_CASE(GaussianDistributionDimensionalityConstructor)
{
  GaussianDistribution d(4);

  BOOST_REQUIRE_EQUAL(d.Mean().n_elem, 4);
  BOOST_REQUIRE_EQUAL(d.Covariance().n_rows, 4);
  BOOST_REQUIRE_EQUAL(d.Covariance().n_cols, 4);
}

/**
 * Make sure Gaussian distributions are initialized correctly when we give a
 * mean and covariance.
 */
BOOST_AUTO_TEST_CASE(GaussianDistributionDistributionConstructor)
{
  arma::vec mean(3);
  arma::mat covariance(3, 3);

  mean.randu();
  covariance.randu();
  covariance *= covariance.t();
  covariance += arma::eye<arma::mat>(3, 3);

  GaussianDistribution d(mean, covariance);

  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_CLOSE(d.Mean()[i], mean[i], 1e-5);

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      BOOST_REQUIRE_CLOSE(d.Covariance()(i, j), covariance(i, j), 1e-5);
}

/**
 * Make sure the probability of observations is correct.
 */
BOOST_AUTO_TEST_CASE(GaussianDistributionProbabilityTest)
{
  arma::vec mean("5 6 3 3 2");
  arma::mat cov("6 1 1 1 2;"
                "1 7 1 0 0;"
                "1 1 4 1 1;"
                "1 0 1 7 0;"
                "2 0 1 0 6");

  GaussianDistribution d(mean, cov);

  BOOST_REQUIRE_CLOSE(d.LogProbability("0 1 2 3 4"), -13.432076798791542, 1e-5);
  BOOST_REQUIRE_CLOSE(d.LogProbability("3 2 3 7 8"), -15.814880322345738, 1e-5);
  BOOST_REQUIRE_CLOSE(d.LogProbability("2 2 0 8 1"), -13.754462857772776, 1e-5);
  BOOST_REQUIRE_CLOSE(d.LogProbability("2 1 5 0 1"), -13.283283233107898, 1e-5);
  BOOST_REQUIRE_CLOSE(d.LogProbability("3 0 5 1 0"), -13.800326511545279, 1e-5);
  BOOST_REQUIRE_CLOSE(d.LogProbability("4 0 6 1 0"), -14.900192463287908, 1e-5);
}

/**
 * Test GaussianDistribution::Probability() in the univariate case.
 */
BOOST_AUTO_TEST_CASE(GaussianUnivariateProbabilityTest)
{
  GaussianDistribution g(arma::vec("0.0"), arma::mat("1.0"));

  // Simple case.
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("0.0")), 0.398942280401433, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("1.0")), 0.241970724519143, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("-1.0")), 0.241970724519143,
      1e-5);

  // A few more cases...
  arma::mat covariance;

  covariance = 2.0;
  g.Covariance(std::move(covariance));
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("0.0")), 0.282094791773878, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("1.0")), 0.219695644733861, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("-1.0")), 0.219695644733861,
      1e-5);

  g.Mean().fill(1.0);
  covariance = 1.0;
  g.Covariance(std::move(covariance));
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("1.0")), 0.398942280401433, 1e-5);

  covariance = 2.0;
  g.Covariance(std::move(covariance));
  BOOST_REQUIRE_CLOSE(g.Probability(arma::vec("-1.0")), 0.103776874355149,
      1e-5);
}

/**
 * Test GaussianDistribution::Probability() in the multivariate case.
 */
BOOST_AUTO_TEST_CASE(GaussianMultivariateProbabilityTest)
{
  // Simple case.
  arma::vec mean = "0 0";
  arma::mat cov = "1 0; 0 1";
  arma::vec x = "0 0";

  GaussianDistribution g(mean, cov);

  BOOST_REQUIRE_CLOSE(g.Probability(x), 0.159154943091895, 1e-5);

  arma::mat covariance;
  covariance = "2 0; 0 2";
  g.Covariance(std::move(covariance));

  BOOST_REQUIRE_CLOSE(g.Probability(x), 0.0795774715459477, 1e-5);

  x = "1 1";

  BOOST_REQUIRE_CLOSE(g.Probability(x), 0.0482661763150270, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(-x), 0.0482661763150270, 1e-5);

  g.Mean() = "1 1";
  BOOST_REQUIRE_CLOSE(g.Probability(x), 0.0795774715459477, 1e-5);
  g.Mean() *= -1;
  BOOST_REQUIRE_CLOSE(g.Probability(-x), 0.0795774715459477, 1e-5);

  g.Mean() = "1 1";
  covariance = "2 1.5; 1.5 4";
  g.Covariance(std::move(covariance));

  BOOST_REQUIRE_CLOSE(g.Probability(x), 0.066372199406187285, 1e-5);
  g.Mean() *= -1;
  BOOST_REQUIRE_CLOSE(g.Probability(-x), 0.066372199406187285, 1e-5);

  g.Mean() = "1 1";
  x = "-1 4";

  BOOST_REQUIRE_CLOSE(g.Probability(x), 0.00072147262356379415, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(-x), 0.00085851785428674523, 1e-5);

  // Higher-dimensional case.
  x = "0 1 2 3 4";
  g.Mean() = "5 6 3 3 2";

  covariance = "6 1 1 1 2;"
               "1 7 1 0 0;"
               "1 1 4 1 1;"
               "1 0 1 7 0;"
               "2 0 1 0 6";
  g.Covariance(std::move(covariance));

  BOOST_REQUIRE_CLOSE(g.Probability(x), 1.4673143531128877e-06, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(-x), 7.7404143494891786e-09, 1e-8);

  g.Mean() *= -1;
  BOOST_REQUIRE_CLOSE(g.Probability(-x), 1.4673143531128877e-06, 1e-5);
  BOOST_REQUIRE_CLOSE(g.Probability(x), 7.7404143494891786e-09, 1e-8);

}

/**
 * Test the phi() function, for multiple points in the multivariate Gaussian
 * case.
 */
BOOST_AUTO_TEST_CASE(GaussianMultipointMultivariateProbabilityTest)
{
  // Same case as before.
  arma::vec mean = "5 6 3 3 2";
  arma::mat cov("6 1 1 1 2;"
                "1 7 1 0 0;"
                "1 1 4 1 1;"
                "1 0 1 7 0;"
                "2 0 1 0 6");

  arma::mat points = "0 3 2 2 3 4;"
                     "1 2 2 1 0 0;"
                     "2 3 0 5 5 6;"
                     "3 7 8 0 1 1;"
                     "4 8 1 1 0 0;";

  arma::vec phis;
  GaussianDistribution g(mean, cov);
  g.LogProbability(points, phis);

  BOOST_REQUIRE_EQUAL(phis.n_elem, 6);

  BOOST_REQUIRE_CLOSE(phis(0), -13.432076798791542, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(1), -15.814880322345738, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(2), -13.754462857772776, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(3), -13.283283233107898, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(4), -13.800326511545279, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(5), -14.900192463287908, 1e-5);
}

/**
 * Make sure random observations follow the probability distribution correctly.
 */
BOOST_AUTO_TEST_CASE(GaussianDistributionRandomTest)
{
  arma::vec mean("1.0 2.25");
  arma::mat cov("0.85 0.60;"
                "0.60 1.45");

  GaussianDistribution d(mean, cov);

  arma::mat obs(2, 5000);

  for (size_t i = 0; i < 5000; i++)
    obs.col(i) = d.Random();

  // Now make sure that reflects the actual distribution.
  arma::vec obsMean = arma::mean(obs, 1);
  arma::mat obsCov = ccov(obs);

  // 10% tolerance because this can be noisy.
  BOOST_REQUIRE_CLOSE(obsMean[0], mean[0], 10.0);
  BOOST_REQUIRE_CLOSE(obsMean[1], mean[1], 10.0);

  BOOST_REQUIRE_CLOSE(obsCov(0, 0), cov(0, 0), 10.0);
  BOOST_REQUIRE_CLOSE(obsCov(0, 1), cov(0, 1), 10.0);
  BOOST_REQUIRE_CLOSE(obsCov(1, 0), cov(1, 0), 10.0);
  BOOST_REQUIRE_CLOSE(obsCov(1, 1), cov(1, 1), 10.0);
}

/**
 * Make sure that we can properly estimate from given observations.
 */
BOOST_AUTO_TEST_CASE(GaussianDistributionTrainTest)
{
  arma::vec mean("1.0 3.0 0.0 2.5");
  arma::mat cov("3.0 0.0 1.0 4.0;"
                "0.0 2.4 0.5 0.1;"
                "1.0 0.5 6.3 0.0;"
                "4.0 0.1 0.0 9.1");

  // Now generate the observations.
  arma::mat observations(4, 10000);

  arma::mat transChol = trans(chol(cov));
  for (size_t i = 0; i < 10000; i++)
    observations.col(i) = transChol * arma::randn<arma::vec>(4) + mean;

  // Now estimate.
  GaussianDistribution d;

  // Find actual mean and covariance of data.
  arma::vec actualMean = arma::mean(observations, 1);
  arma::mat actualCov = ccov(observations);

  d.Train(observations);

  // Check that everything is estimated right.
  for (size_t i = 0; i < 4; i++)
    BOOST_REQUIRE_SMALL(d.Mean()[i] - actualMean[i], 1e-5);

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      BOOST_REQUIRE_SMALL(d.Covariance()(i, j) - actualCov(i, j), 1e-5);
}

/******************************/
/** Gamma Distribution Tests **/
/******************************/

/**
 * Make sure that using an object to fit one reference set and then asking
 * to fit another works properly.
 */
BOOST_AUTO_TEST_CASE(GammaDistributionTrainTest)
{
  // Create a gamma distribution random generator.
  double alphaReal = 5.3;
  double betaReal = 1.5;
  std::default_random_engine generator;
  std::gamma_distribution<double> dist(alphaReal, betaReal);

  // Create a N x d gamma distribution data and fit the results.
  size_t N = 200;
  size_t d = 2;
  arma::mat rdata(d, N);

  // Random generation of gamma-like points.
  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata(j, i) = dist(generator);

  // Create Gamma object and call Train() on reference set.
  GammaDistribution gDist;
  gDist.Train(rdata);

  // Training must estimate d pairs of alpha and beta parameters.
  BOOST_REQUIRE_EQUAL(gDist.Dimensionality(), d);
  BOOST_REQUIRE_EQUAL(gDist.Dimensionality(), d);

  // Create a N' x d' gamma distribution, fit results without new object.
  size_t N2 = 350;
  size_t d2 = 4;
  arma::mat rdata2(d2, N2);

  // Random generation of gamma-like points.
  for (size_t j = 0; j < d2; ++j)
    for (size_t i = 0; i < N2; ++i)
      rdata2(j, i) = dist(generator);

  // Fit results using old object.
  gDist.Train(rdata2);

  // Training must estimate d' pairs of alpha and beta parameters.
  BOOST_REQUIRE_EQUAL(gDist.Dimensionality(), d2);
  BOOST_REQUIRE_EQUAL(gDist.Dimensionality(), d2);
}

/**
 * This test verifies that the fitting procedure for GammaDistribution works
 * properly and converges near the actual gamma parameters. We do this twice
 * with different alpha/beta parameters so we make sure we don't have some weird
 * bug that always converges to the same number.
 */
BOOST_AUTO_TEST_CASE(GammaDistributionFittingTest)
{
  // Offset from the actual alpha/beta. 10% is quite a relaxed tolerance since
  // the random points we generate are few (for test speed) and might be fitted
  // better by a similar distribution.
  double errorTolerance = 10;

  size_t N = 500;
  size_t d = 1; // Only 1 dimension is required for this.

  /** Iteration 1 (first parameter set) **/

  // Create a gamma-random generator and data
  double alphaReal = 5.3;
  double betaReal = 1.5;
  std::default_random_engine generator;
  std::gamma_distribution<double> dist(alphaReal, betaReal);

  // Random generation of gamma-like points.
  arma::mat rdata(d, N);
  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata(j, i) = dist(generator);

  // Create Gamma object and call Train() on reference set.
  GammaDistribution gDist;
  gDist.Train(rdata);

  // Estimated parameter must be close to real.
  BOOST_REQUIRE_CLOSE(gDist.Alpha(0), alphaReal, errorTolerance);
  BOOST_REQUIRE_CLOSE(gDist.Beta(0), betaReal, errorTolerance);

  /** Iteration 2 (different parameter set) **/

  // Create a gamma-random generator and data
  double alphaReal2 = 7.2;
  double betaReal2 = 0.9;
  std::default_random_engine generator2;
  std::gamma_distribution<double> dist2(alphaReal2, betaReal2);

  // Random generation of gamma-like points.
  arma::mat rdata2(d, N);
  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata2(j, i) = dist2(generator2);

  // Create Gamma object and call Train() on reference set.
  GammaDistribution gDist2;
  gDist2.Train(rdata2);

  // Estimated parameter must be close to real.
  BOOST_REQUIRE_CLOSE(gDist2.Alpha(0), alphaReal2, errorTolerance);
  BOOST_REQUIRE_CLOSE(gDist2.Beta(0), betaReal2, errorTolerance);
}

/**
 * Test that Train() and the constructor that takes data give the same resulting
 * distribution.
 */
BOOST_AUTO_TEST_CASE(GammaDistributionTrainConstructorTest)
{
  const arma::mat data = arma::randu<arma::mat>(10, 500);

  GammaDistribution d1(data);
  GammaDistribution d2;
  d2.Train(data);

  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_CLOSE(d1.Alpha(i), d2.Alpha(i), 1e-5);
    BOOST_REQUIRE_CLOSE(d1.Beta(i), d2.Beta(i), 1e-5);
  }
}

/**
 * Test that Train() with a dataset and Train() with dataset statistics return
 * the same results.
 */
BOOST_AUTO_TEST_CASE(GammaDistributionTrainStatisticsTest)
{
  const arma::mat data = arma::randu<arma::mat>(1, 500);

  // Train object d1 with the data.
  GammaDistribution d1(data);

  // Train object d2 with the data's statistics.
  GammaDistribution d2;
  const arma::vec meanLogx = arma::mean(arma::log(data), 1);
  const arma::vec meanx = arma::mean(data, 1);
  const arma::vec logMeanx = arma::log(meanx);
  d2.Train(logMeanx, meanLogx, meanx);

  BOOST_REQUIRE_CLOSE(d1.Alpha(0), d2.Alpha(0), 1e-5);
  BOOST_REQUIRE_CLOSE(d1.Beta(0), d2.Beta(0), 1e-5);
}

/**
 * Tests that Random() generates points that can be reasonably well fit by the
 * distribution that generated them.
 */
BOOST_AUTO_TEST_CASE(GammaDistributionRandomTest)
{
  const arma::vec a("2.0 2.5 3.0"), b("0.4 0.6 1.3");
  const size_t numPoints = 2000;

  // Distribution to generate points.
  GammaDistribution d1(a, b);
  arma::mat data(3, numPoints); // 3-d points.

  for (size_t i = 0; i < numPoints; ++i)
    //std::cout << d1.Random() << "====" << std::endl;
    data.col(i) = d1.Random();
  
  // Distribution to fit points.
  GammaDistribution d2(data);
  for (size_t i = 0; i < 3; ++i)
  {
    BOOST_REQUIRE_CLOSE(d2.Alpha(i), a(i), 10); // Within 10%
    BOOST_REQUIRE_CLOSE(d2.Beta(i), b(i), 10);
  }
}

BOOST_AUTO_TEST_CASE(GammaDistributionProbabilityTest)
{
  // Train two 1-dimensional distributions.
  const arma::vec a1("2.0"), b1("0.9"), a2("3.1"), b2("1.4");
  arma::mat x1("2.0"), x2("2.94");
  arma::vec prob1, prob2;

  // Evaluated at wolfram|alpha
  GammaDistribution d1(a1, b1);
  d1.Probability(x1, prob1);
  BOOST_REQUIRE_CLOSE(prob1(0), 0.267575, 1e-3);

  // Evaluated at wolfram|alpha
  GammaDistribution d2(a2, b2);
  d2.Probability(x2, prob2);
  BOOST_REQUIRE_CLOSE(prob2(0), 0.189043, 1e-3);

  // Check that the overload that returns the probability for 1 dimension
  // agrees.
  BOOST_REQUIRE_CLOSE(prob2(0), d2.Probability(2.94, 0), 1e-5);

  // Combine into one 2-dimensional distribution.
  const arma::vec a3("2.0 3.1"), b3("0.9 1.4");
  arma::mat x3(2, 2);
  x3
    << 2.0 << 2.94 << arma::endr
    << 2.0 << 2.94;
  arma::vec prob3;

  // Expect that the 2-dimensional distribution returns the product of the
  // 1-dimensional distributions (evaluated at wolfram|alpha).
  GammaDistribution d3(a3, b3);
  d3.Probability(x3, prob3);
  BOOST_REQUIRE_CLOSE(prob3(0), 0.04408, 1e-2);
  BOOST_REQUIRE_CLOSE(prob3(1), 0.026165, 1e-2);
}

BOOST_AUTO_TEST_CASE(GammaDistributionLogProbabilityTest)
{
  // Train two 1-dimensional distributions.
  const arma::vec a1("2.0"), b1("0.9"), a2("3.1"), b2("1.4");
  arma::mat x1("2.0"), x2("2.94");
  arma::vec prob1, prob2;

  // Evaluated at wolfram|alpha
  GammaDistribution d1(a1, b1);
  d1.LogProbability(x1, prob1);
  BOOST_REQUIRE_CLOSE(prob1(0), std::log(0.267575), 1e-3);

  // Evaluated at wolfram|alpha
  GammaDistribution d2(a2, b2);
  d2.LogProbability(x2, prob2);
  BOOST_REQUIRE_CLOSE(prob2(0), std::log(0.189043), 1e-3);

  // Combine into one 2-dimensional distribution.
  const arma::vec a3("2.0 3.1"), b3("0.9 1.4");
  arma::mat x3(2, 2);
  x3
    << 2.0 << 2.94 << arma::endr
    << 2.0 << 2.94;
  arma::vec prob3;

  // Expect that the 2-dimensional distribution returns the product of the
  // 1-dimensional distributions (evaluated at wolfram|alpha).
  GammaDistribution d3(a3, b3);
  d3.LogProbability(x3, prob3);
  BOOST_REQUIRE_CLOSE(prob3(0), std::log(0.04408), 1e-3);
  BOOST_REQUIRE_CLOSE(prob3(1), std::log(0.026165), 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();
