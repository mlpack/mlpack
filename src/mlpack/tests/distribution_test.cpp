/**
 * @file distribution_test.cpp
 * @author Ryan Curtin
 * @author Yannis Mentekidis
 *
 * Tests for the classes:
 *  * mlpack::distribution::DiscreteDistribution
 *  * mlpack::distribution::GaussianDistribution
 *  * mlpack::distribution::GammaDistribution
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
 * Make sure that both the default constructor and parameterized constructor
 * create the GammaDistribution object properly.
 */
BOOST_AUTO_TEST_CASE(GammaDistributionConstructorTest)
{
  // Empty constructor test. Make sure reference set is empty.
  GammaDistribution gDist;
  BOOST_REQUIRE_EQUAL(arma::size(gDist.ReferenceSet()),
      arma::size(arma::mat()));

  // Parameterized constructor test. Make sure reference set is passed
  // correctly.
  // Create an Nxd reference set
  size_t N = 200;
  size_t d = 3;
  arma::mat rdata(d, N);
  
  double alphaReal = 5.3;
  double betaReal = 1.5;

  // Create gamma distribution.
  std::default_random_engine generator;
  std::gamma_distribution<double> dist(alphaReal, betaReal);

  // Random generation of gamma-like points.
  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata(j, i) = dist(generator);

  // Create Gamma object.
  GammaDistribution gDist2(rdata);
  BOOST_REQUIRE_EQUAL(arma::size(gDist2.ReferenceSet()),
      arma::size(arma::mat(d, N)));
}

/**
 * Make sure that constructing an object with one reference set and then asking
 * to fit another works properly: The object retains the old reference set, but
 * fits parameters (alpha, beta) to the new reference set.
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
  GammaDistribution gDist(rdata);
  gDist.Train();

  // Training must estimate d pairs of alpha and beta parameters.
  BOOST_REQUIRE_EQUAL(gDist.Alpha().n_elem, d);
  BOOST_REQUIRE_EQUAL(gDist.Beta().n_elem, d);

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
  BOOST_REQUIRE_EQUAL(gDist.Alpha().n_elem, d2);
  BOOST_REQUIRE_EQUAL(gDist.Beta().n_elem, d2);
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
  GammaDistribution gDist(rdata);
  gDist.Train();

  // Estimated parameter must be close to real.
  BOOST_REQUIRE_CLOSE(gDist.Alpha()[0], alphaReal, errorTolerance);
  BOOST_REQUIRE_CLOSE(gDist.Beta()[0], betaReal, errorTolerance);

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
  GammaDistribution gDist2(rdata2);
  gDist2.Train();

  // Estimated parameter must be close to real.
  BOOST_REQUIRE_CLOSE(gDist2.Alpha()[0], alphaReal2, errorTolerance);
  BOOST_REQUIRE_CLOSE(gDist2.Beta()[0], betaReal2, errorTolerance);
}

BOOST_AUTO_TEST_SUITE_END();
