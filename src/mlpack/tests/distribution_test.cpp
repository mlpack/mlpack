/**
 * @file tests/distribution_test.cpp
 * @author Ryan Curtin
 * @author Yannis Mentekidis
 * @author Rohan Raj
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

#include <mlpack/core/dists/regression_distribution.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>

#include "catch.hpp"
#include "serialization_catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace mlpack::distribution;
using namespace mlpack::metric;
using namespace mlpack::math;

/*********************************/
/** Discrete Distribution Tests **/
/*********************************/

/**
 * Make sure we initialize correctly.
 */
TEST_CASE("DiscreteDistributionConstructorTest", "[DistributionTest]")
{
  DiscreteDistribution d(5);

  REQUIRE(d.Probabilities().n_elem == 5);
  REQUIRE(d.Probability("0") == Approx(0.2).epsilon(1e-7));
  REQUIRE(d.Probability("1") == Approx(0.2).epsilon(1e-7));
  REQUIRE(d.Probability("2") == Approx(0.2).epsilon(1e-7));
  REQUIRE(d.Probability("3") == Approx(0.2).epsilon(1e-7));
  REQUIRE(d.Probability("4") == Approx(0.2).epsilon(1e-7));
}

/**
 * Make sure we get the probabilities of observations right.
 */
TEST_CASE("DiscreteDistributionProbabilityTest", "[DistributionTest]")
{
  DiscreteDistribution d(5);

  d.Probabilities() = "0.2 0.4 0.1 0.1 0.2";

  REQUIRE(d.Probability("0") == Approx(0.2).epsilon(1e-7));
  REQUIRE(d.Probability("1") == Approx(0.4).epsilon(1e-7));
  REQUIRE(d.Probability("2") == Approx(0.1).epsilon(1e-7));
  REQUIRE(d.Probability("3") == Approx(0.1).epsilon(1e-7));
  REQUIRE(d.Probability("4") == Approx(0.2).epsilon(1e-7));
}

/**
 * Make sure we get random observations correct.
 */
TEST_CASE("DiscreteDistributionRandomTest", "[DistributionTest]")
{
  DiscreteDistribution d(arma::Col<size_t>("3"));

  d.Probabilities() = "0.3 0.6 0.1";

  arma::vec actualProb(3);

  actualProb.zeros();

  for (size_t i = 0; i < 50000; ++i)
    actualProb((size_t) (d.Random()[0] + 0.5))++;

  // Normalize.
  actualProb /= accu(actualProb);

  // 8% tolerance, because this can be a noisy process.
  REQUIRE(actualProb(0) == Approx(0.3).epsilon(0.08));
  REQUIRE(actualProb(1) == Approx(0.6).epsilon(0.08));
  REQUIRE(actualProb(2) == Approx(0.1).epsilon(0.08));
}

/**
 * Make sure we can estimate from observations correctly.
 */
TEST_CASE("DiscreteDistributionTrainTest", "[DistributionTest]")
{
  DiscreteDistribution d(4);

  arma::mat obs("0 0 1 1 2 2 2 3");

  d.Train(obs);

  REQUIRE(d.Probability("0") == Approx(0.25).epsilon(1e-7));
  REQUIRE(d.Probability("1") == Approx(0.25).epsilon(1e-7));
  REQUIRE(d.Probability("2") == Approx(0.375).epsilon(1e-7));
  REQUIRE(d.Probability("3") == Approx(0.125).epsilon(1e-7));
}

/**
 * Estimate from observations with probabilities.
 */
TEST_CASE("DiscreteDistributionTrainProbTest", "[DistributionTest]")
{
  DiscreteDistribution d(3);

  arma::mat obs("0 0 1 2");

  arma::vec prob("0.25 0.25 0.5 1.0");

  d.Train(obs, prob);

  REQUIRE(d.Probability("0") == Approx(0.25).epsilon(1e-7));
  REQUIRE(d.Probability("1") == Approx(0.25).epsilon(1e-7));
  REQUIRE(d.Probability("2") == Approx(0.5).epsilon(1e-7));
}

/**
 * Achieve multidimensional probability distribution.
 */
TEST_CASE("MultiDiscreteDistributionTrainProbTest", "[DistributionTest]")
{
  DiscreteDistribution d("10 10 10");

  arma::mat obs("0 1 1 1 2 2 2 2 2 2;"
                "0 0 0 1 1 1 2 2 2 2;"
                "0 0 0 1 1 2 2 2 2 2;");

  d.Train(obs);
  REQUIRE(d.Probability("0 0 0") == Approx(0.009).epsilon(1e-7));
  REQUIRE(d.Probability("0 1 2") == Approx(0.015).epsilon(1e-7));
  REQUIRE(d.Probability("2 1 0") == Approx(0.054).epsilon(1e-7));
}

/**
 * Make sure we initialize multidimensional probability distribution
 * correctly.
 */
TEST_CASE("MultiDiscreteDistributionConstructorTest", "[DistributionTest]")
{
  DiscreteDistribution d("4 4 4 4");

  REQUIRE(d.Probabilities(0).size() == 4);
  REQUIRE(d.Dimensionality() == 4);
  REQUIRE(d.Probability("0 0 0 0") == Approx(0.00390625).epsilon(1e-7));
  REQUIRE(d.Probability("0 1 2 3") == Approx(0.00390625).epsilon(1e-7));
}

/**
 * Achieve multidimensional probability distribution.
 */
TEST_CASE("MultiDiscreteDistributionTrainTest", "[DistributionTest]")
{
  std::vector<arma::vec> pro;
  pro.push_back(arma::vec("0.1, 0.3, 0.6"));
  pro.push_back(arma::vec("0.3, 0.3, 0.3"));
  pro.push_back(arma::vec("0.25, 0.25, 0.5"));

  DiscreteDistribution d(pro);

  REQUIRE(d.Probability("0 0 0") == Approx(0.0083333).epsilon(1e-5));
  REQUIRE(d.Probability("0 1 2") == Approx(0.0166666).epsilon(1e-5));
  REQUIRE(d.Probability("2 1 0") == Approx(0.05).epsilon(1e-7));
}

/**
 * Estimate multidimensional probability distribution from observations with
 * probabilities.
 */
TEST_CASE("MultiDiscreteDistributionTrainProTest", "[DistributionTest]")
{
  DiscreteDistribution d("5 5 5");

  arma::mat obs("0 0 1 1 2;"
                "0 1 1 2 2;"
                "0 1 1 2 2");

  arma::vec prob("0.25 0.25 0.25 0.25 1");

  d.Train(obs, prob);

  REQUIRE(d.Probability("0 0 0") == Approx(0.00390625).epsilon(1e-7));
  REQUIRE(d.Probability("1 0 1") == Approx(0.0078125).epsilon(1e-7));
  REQUIRE(d.Probability("2 1 0") == Approx(0.015625).epsilon(1e-7));
}

/**
 * Test the LogProbability() function, for multiple points in the multivariate
 * Discrete case.
 */
TEST_CASE("DiscreteLogProbabilityTest", "[DistributionTest]")
{
  // Same case as before.
  DiscreteDistribution d("5 5");

  arma::mat obs("0 2;"
                "1 2;");

  arma::vec logProb;

  d.LogProbability(obs, logProb);

  REQUIRE(logProb.n_elem == 2);

  REQUIRE(logProb(0) == Approx(-3.2188758248682).epsilon(1e-5));
  REQUIRE(logProb(1) == Approx(-3.2188758248682).epsilon(1e-5));
}

/**
 * Test the Probability() function, for multiple points in the multivariate
 * Discrete case.
 */
TEST_CASE("DiscreteProbabilityTest", "[DistributionTest]")
{
  // Same case as before.
  DiscreteDistribution d("5 5");

  arma::mat obs("0 2;"
                "1 2;");

  arma::vec prob;

  d.Probability(obs, prob);

  REQUIRE(prob.n_elem == 2);

  REQUIRE(prob(0) == Approx(0.0400000000000).epsilon(1e-5));
  REQUIRE(prob(1) == Approx(0.0400000000000).epsilon(1e-5));
}

/*********************************/
/** Gaussian Distribution Tests **/
/*********************************/

/**
 * Make sure Gaussian distributions are initialized correctly.
 */
TEST_CASE("GaussianDistributionEmptyConstructor", "[DistributionTest]")
{
  GaussianDistribution d;

  REQUIRE(d.Mean().n_elem == 0);
  REQUIRE(d.Covariance().n_elem == 0);
}

/**
 * Make sure Gaussian distributions are initialized to the correct
 * dimensionality.
 */
TEST_CASE("GaussianDistributionDimensionalityConstructor",
          "[DistributionTest]")
{
  GaussianDistribution d(4);

  REQUIRE(d.Mean().n_elem == 4);
  REQUIRE(d.Covariance().n_rows == 4);
  REQUIRE(d.Covariance().n_cols == 4);
}

/**
 * Make sure Gaussian distributions are initialized correctly when we give a
 * mean and covariance.
 */
TEST_CASE("GaussianDistributionDistributionConstructor", "[DistributionTest]")
{
  arma::vec mean(3);
  arma::mat covariance(3, 3);

  mean.randu();
  covariance.randu();
  covariance *= covariance.t();
  covariance += arma::eye<arma::mat>(3, 3);

  GaussianDistribution d(mean, covariance);

  for (size_t i = 0; i < 3; ++i)
    REQUIRE(d.Mean()[i] == Approx(mean[i]).epsilon(1e-7));

  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE(d.Covariance()(i, j) == Approx(covariance(i, j)).epsilon(1e-7));
}

/**
 * Make sure the probability of observations is correct.
 */
TEST_CASE("GaussianDistributionProbabilityTest", "[DistributionTest]")
{
  arma::vec mean("5 6 3 3 2");
  arma::mat cov("6 1 1 1 2;"
                "1 7 1 0 0;"
                "1 1 4 1 1;"
                "1 0 1 7 0;"
                "2 0 1 0 6");

  GaussianDistribution d(mean, cov);

  REQUIRE(d.LogProbability("0 1 2 3 4") ==
      Approx(-13.432076798791542).epsilon(1e-7));
  REQUIRE(d.LogProbability("3 2 3 7 8") ==
      Approx(-15.814880322345738).epsilon(1e-7));
  REQUIRE(d.LogProbability("2 2 0 8 1") ==
      Approx(-13.754462857772776).epsilon(1e-7));
  REQUIRE(d.LogProbability("2 1 5 0 1") ==
      Approx(-13.283283233107898).epsilon(1e-7));
  REQUIRE(d.LogProbability("3 0 5 1 0") ==
      Approx(-13.800326511545279).epsilon(1e-7));
  REQUIRE(d.LogProbability("4 0 6 1 0") ==
      Approx(-14.900192463287908).epsilon(1e-7));
}

/**
 * Test GaussianDistribution::Probability() in the univariate case.
 */
TEST_CASE("GaussianUnivariateProbabilityTest", "[DistributionTest]")
{
  GaussianDistribution g(arma::vec("0.0"), arma::mat("1.0"));

  // Simple case.
  REQUIRE(g.Probability(arma::vec("0.0")) ==
      Approx(0.398942280401433).epsilon(1e-7));
  REQUIRE(g.Probability(arma::vec("1.0")) ==
      Approx(0.241970724519143).epsilon(1e-7));
  REQUIRE(g.Probability(arma::vec("-1.0")) ==
      Approx(0.241970724519143).epsilon(1e-7));

  // A few more cases...
  arma::mat covariance;

  covariance = 2.0;
  g.Covariance(std::move(covariance));
  REQUIRE(g.Probability(arma::vec("0.0")) ==
      Approx(0.282094791773878).epsilon(1e-7));
  REQUIRE(g.Probability(arma::vec("1.0")) ==
      Approx(0.219695644733861).epsilon(1e-7));
  REQUIRE(g.Probability(arma::vec("-1.0")) ==
      Approx(0.219695644733861).epsilon(1e-7));

  g.Mean().fill(1.0);
  covariance = 1.0;
  g.Covariance(std::move(covariance));
  REQUIRE(g.Probability(arma::vec("1.0")) ==
      Approx(0.398942280401433).epsilon(1e-7));

  covariance = 2.0;
  g.Covariance(std::move(covariance));
  REQUIRE(g.Probability(arma::vec("-1.0")) ==
      Approx(0.103776874355149).epsilon(1e-7));
}

/**
 * Test GaussianDistribution::Probability() in the multivariate case.
 */
TEST_CASE("GaussianMultivariateProbabilityTest", "[DistributionTest]")
{
  // Simple case.
  arma::vec mean = "0 0";
  arma::mat cov = "1 0; 0 1";
  arma::vec x = "0 0";

  GaussianDistribution g(mean, cov);

  REQUIRE(g.Probability(x) == Approx(0.159154943091895).epsilon(1e-7));

  arma::mat covariance;
  covariance = "2 0; 0 2";
  g.Covariance(std::move(covariance));

  REQUIRE(g.Probability(x) == Approx(0.0795774715459477).epsilon(1e-7));

  x = "1 1";

  REQUIRE(g.Probability(x) == Approx(0.0482661763150270).epsilon(1e-7));
  REQUIRE(g.Probability(-x) == Approx(0.0482661763150270).epsilon(1e-7));

  g.Mean() = "1 1";
  REQUIRE(g.Probability(x) == Approx(0.0795774715459477).epsilon(1e-7));
  g.Mean() *= -1;
  REQUIRE(g.Probability(-x) == Approx(0.0795774715459477).epsilon(1e-7));

  g.Mean() = "1 1";
  covariance = "2 1.5; 1.5 4";
  g.Covariance(std::move(covariance));

  REQUIRE(g.Probability(x) == Approx(0.066372199406187285).epsilon(1e-7));
  g.Mean() *= -1;
  REQUIRE(g.Probability(-x) == Approx(0.066372199406187285).epsilon(1e-7));

  g.Mean() = "1 1";
  x = "-1 4";

  REQUIRE(g.Probability(x) == Approx(0.00072147262356379415).epsilon(1e-7));
  REQUIRE(g.Probability(-x) == Approx(0.00085851785428674523).epsilon(1e-7));

  // Higher-dimensional case.
  x = "0 1 2 3 4";
  g.Mean() = "5 6 3 3 2";

  covariance = "6 1 1 1 2;"
               "1 7 1 0 0;"
               "1 1 4 1 1;"
               "1 0 1 7 0;"
               "2 0 1 0 6";
  g.Covariance(std::move(covariance));

  REQUIRE(g.Probability(x) == Approx(1.4673143531128877e-06).epsilon(1e-7));
  REQUIRE(g.Probability(-x) == Approx(7.7404143494891786e-09).epsilon(1e-10));

  g.Mean() *= -1;
  REQUIRE(g.Probability(-x) == Approx(1.4673143531128877e-06).epsilon(1e-7));
  REQUIRE(g.Probability(x) == Approx(7.7404143494891786e-09).epsilon(1e-10));
}

/**
 * Test the phi() function, for multiple points in the multivariate Gaussian
 * case.
 */
TEST_CASE("GaussianMultipointMultivariateProbabilityTest", "[DistributionTest]")
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

  REQUIRE(phis.n_elem == 6);

  REQUIRE(phis(0) == Approx(-13.432076798791542).epsilon(1e-7));
  REQUIRE(phis(1) == Approx(-15.814880322345738).epsilon(1e-7));
  REQUIRE(phis(2) == Approx(-13.754462857772776).epsilon(1e-7));
  REQUIRE(phis(3) == Approx(-13.283283233107898).epsilon(1e-7));
  REQUIRE(phis(4) == Approx(-13.800326511545279).epsilon(1e-7));
  REQUIRE(phis(5) == Approx(-14.900192463287908).epsilon(1e-7));
}

/**
 * Make sure random observations follow the probability distribution correctly.
 */
TEST_CASE("GaussianDistributionRandomTest", "[DistributionTest]")
{
  arma::vec mean("1.0 2.25");
  arma::mat cov("0.85 0.60;"
                "0.60 1.45");

  GaussianDistribution d(mean, cov);

  arma::mat obs(2, 5000);

  for (size_t i = 0; i < 5000; ++i)
    obs.col(i) = d.Random();

  // Now make sure that reflects the actual distribution.
  arma::vec obsMean = arma::mean(obs, 1);
  arma::mat obsCov = mlpack::math::ColumnCovariance(obs);

  // 10% tolerance because this can be noisy.
  REQUIRE(obsMean[0] == Approx(mean[0]).epsilon(0.1));
  REQUIRE(obsMean[1] == Approx(mean[1]).epsilon(0.1));

  REQUIRE(obsCov(0, 0) == Approx(cov(0, 0)).epsilon(0.1));
  REQUIRE(obsCov(0, 1) == Approx(cov(0, 1)).epsilon(0.1));
  REQUIRE(obsCov(1, 0) == Approx(cov(1, 0)).epsilon(0.1));
  REQUIRE(obsCov(1, 1) == Approx(cov(1, 1)).epsilon(0.1));
}

/**
 * Make sure that we can properly estimate from given observations.
 */
TEST_CASE("GaussianDistributionTrainTest", "[DistributionTest]")
{
  arma::vec mean("1.0 3.0 0.0 2.5");
  arma::mat cov("3.0 0.0 1.0 4.0;"
                "0.0 2.4 0.5 0.1;"
                "1.0 0.5 6.3 0.0;"
                "4.0 0.1 0.0 9.1");

  // Now generate the observations.
  arma::mat observations(4, 10000);

  arma::mat transChol = trans(chol(cov));
  for (size_t i = 0; i < 10000; ++i)
    observations.col(i) = transChol * arma::randn<arma::vec>(4) + mean;

  // Now estimate.
  GaussianDistribution d;

  // Find actual mean and covariance of data.
  arma::vec actualMean = arma::mean(observations, 1);
  arma::mat actualCov = mlpack::math::ColumnCovariance(observations);

  d.Train(observations);

  // Check that everything is estimated right.
  for (size_t i = 0; i < 4; ++i)
    REQUIRE(d.Mean()[i] - actualMean[i] == Approx(0.0).margin(1e-5));

  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 4; ++j)
    {
      REQUIRE(d.Covariance()(i, j) - actualCov(i, j) ==
          Approx(0.0).margin(1e-5));
    }
}

/**
 * This test verifies the fitting of GaussianDistribution works properly when
 * probabilities for each sample is given.
 */
TEST_CASE("GaussianDistributionTrainWithProbabilitiesTest",
          "[DistributionTest]")
{
  arma::vec mean = ("5.0");
  arma::vec cov = ("2.0");

  GaussianDistribution dist(mean, cov);
  size_t N = 5000;
  size_t d = 1;

  arma::mat rdata(d, N);
  for (size_t i = 0; i < N; ++i)
    rdata.col(i) = dist.Random();

  arma::vec probabilities(N);
  for (size_t i = 0; i < N; ++i)
    probabilities(i) = Random();

  // Fit distribution with probabilities and data.
  GaussianDistribution guDist;
  guDist.Train(rdata, probabilities);

  // Fit distribution only with data.
  GaussianDistribution guDist2;
  guDist2.Train(rdata);

  REQUIRE(guDist.Mean()[0] == Approx(guDist2.Mean()[0]).epsilon(0.06));
  REQUIRE(guDist.Covariance()[0] ==
      Approx(guDist2.Covariance()[0]).epsilon(0.06));

  REQUIRE(guDist.Mean()[0] == Approx(mean[0]).epsilon(0.06));
  REQUIRE(guDist.Covariance()[0] == Approx(cov[0]).epsilon(0.06));
}

/**
 * This test ensures that the same result is obtained when trained with
 * probabilities all set to 1 and with no probabilities at all.
 */
TEST_CASE("GaussianDistributionWithProbabilties1Test", "[DistributionTest]")
{
  arma::vec mean = ("5.0");
  arma::vec cov  = ("4.0");

  GaussianDistribution dist(mean, cov);
  size_t N = 50000;
  size_t d = 1;

  arma::mat rdata(d, N);

  for (size_t i = 0; i < N; ++i)
      rdata.col(i) = Random();

  arma::vec probabilities(N, arma::fill::ones);

  // Fit the distribution with only data.
  GaussianDistribution guDist;
  guDist.Train(rdata);

  // Fit the distribution with data and each probability as 1.
  GaussianDistribution guDist2;
  guDist2.Train(rdata, probabilities);

  REQUIRE(guDist.Mean()[0] == Approx(guDist2.Mean()[0]).epsilon(1e-17));
  REQUIRE(guDist.Covariance()[0] ==
      Approx(guDist2.Covariance()[0]).epsilon(1e-4));
}

/**
 * This test draws points from two different normal distributions, sets the
 * probabilities for points from the first distribution to something small and
 * the probabilities for the second to something large.
 *
 * We expect that the distribution we recover after training to be the same as
 * the second normal distribution (the one with high probabilities).
 */
TEST_CASE("GaussianDistributionTrainWithTwoDistProbabilitiesTest",
          "[DistributionTest]")
{
  arma::vec mean1 = ("5.0");
  arma::vec cov1 = ("4.0");

  arma::vec mean2 = ("3.0");
  arma::vec cov2 = ("1.0");

  // Create two GaussianDistributions with different parameters.
  GaussianDistribution dist1(mean1, cov1);
  GaussianDistribution dist2(mean2, cov2);

  size_t N = 50000;
  size_t d = 1;

  arma::mat rdata(d, N);
  arma::vec probabilities(N);

  // Fill even numbered columns with random points from dist1 and odd numbered
  // columns with random points from dist2.
  for (size_t j = 0; j < N; ++j)
  {
    if (j % 2 == 0)
      rdata.col(j) = dist1.Random();
    else
      rdata.col(j) = dist2.Random();
  }

  // Assign high probabilities to points drawn from dist1 and low probabilities
  // to numbers drawn from dist2.
  for (size_t i = 0 ; i < N ; ++i)
  {
    if (i % 2 == 0)
      probabilities(i) = Random(0.98, 1);
    else
      probabilities(i) = Random(0, 0.02);
  }

  GaussianDistribution guDist;
  guDist.Train(rdata, probabilities);

  REQUIRE(guDist.Mean()[0] == Approx(mean1[0]).epsilon(0.05));
  REQUIRE(guDist.Covariance()[0] == Approx(cov1[0]).epsilon(0.05));
}

/******************************/
/** Gamma Distribution Tests **/
/******************************/
/**
 * Make sure that using an object to fit one reference set and then asking
 * to fit another works properly.
 */
TEST_CASE("GammaDistributionTrainTest", "[DistributionTest]")
{
  // Create a gamma distribution random generator.
  double alphaReal = 5.3;
  double betaReal = 1.5;
  std::gamma_distribution<double> dist(alphaReal, betaReal);

  // Create a N x d gamma distribution data and fit the results.
  size_t N = 200;
  size_t d = 2;
  arma::mat rdata(d, N);

  // Random generation of gamma-like points.
  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata(j, i) = dist(math::randGen);

  // Create Gamma object and call Train() on reference set.
  GammaDistribution gDist;
  gDist.Train(rdata);

  // Training must estimate d pairs of alpha and beta parameters.
  REQUIRE(gDist.Dimensionality() == d);
  REQUIRE(gDist.Dimensionality() == d);

  // Create a N' x d' gamma distribution, fit results without new object.
  size_t N2 = 350;
  size_t d2 = 4;
  arma::mat rdata2(d2, N2);

  // Random generation of gamma-like points.
  for (size_t j = 0; j < d2; ++j)
    for (size_t i = 0; i < N2; ++i)
      rdata2(j, i) = dist(math::randGen);

  // Fit results using old object.
  gDist.Train(rdata2);

  // Training must estimate d' pairs of alpha and beta parameters.
  REQUIRE(gDist.Dimensionality() == d2);
  REQUIRE(gDist.Dimensionality() == d2);
}

/**
 * This test verifies that the fitting procedure for GammaDistribution works
 * properly when probabilities for each sample is given.
 */
TEST_CASE("GammaDistributionTrainWithProbabilitiesTest", "[DistributionTest]")
{
  double alphaReal = 5.4;
  double betaReal = 6.7;

  // Create a gamma distribution random generator.
  std::gamma_distribution<double> dist(alphaReal, betaReal);

  size_t N = 50000;
  size_t d = 2;
  arma::mat rdata(d, N);

  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata(j, i) = dist(math::randGen);

  // Fill the probabilities randomly.
  arma::vec probabilities(N, arma::fill::randu);

  // Fit results with probabilities and data.
  GammaDistribution gDist;
  gDist.Train(rdata, probabilities);

  // Fit results with only data.
  GammaDistribution gDist2;
  gDist2.Train(rdata);

  REQUIRE(gDist2.Alpha(0) == Approx(gDist.Alpha(0)).epsilon(0.015));
  REQUIRE(gDist2.Beta(0) == Approx(gDist.Beta(0)).epsilon(0.015));

  REQUIRE(gDist2.Alpha(1) == Approx(gDist.Alpha(1)).epsilon(0.015));
  REQUIRE(gDist2.Beta(1) == Approx(gDist.Beta(1)).epsilon(0.015));

  REQUIRE(alphaReal == Approx(gDist.Alpha(0)).epsilon(0.03));
  REQUIRE(betaReal == Approx(gDist.Beta(0)).epsilon(0.03));

  REQUIRE(alphaReal == Approx(gDist.Alpha(1)).epsilon(0.03));
  REQUIRE(betaReal == Approx(gDist.Beta(1)).epsilon(0.03));
}

/**
 * This test ensures that the same result is obtained when trained with
 * probabilities all set to 1 and with no probabilities at all.
 */
TEST_CASE("GammaDistributionTrainAllProbabilities1Test", "[DistributionTest]")
{
  double alphaReal = 5.4;
  double betaReal = 6.7;

  // Create a gamma distribution random generator.
  std::gamma_distribution<double> dist(alphaReal, betaReal);

  size_t N = 1000;
  size_t d = 2;
  arma::mat rdata(d, N);

  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata(j, i) = dist(math::randGen);

  // Fit results with only data.
  GammaDistribution gDist;
  gDist.Train(rdata);

  // Fit results with data and each probability as 1.
  GammaDistribution gDist2;
  arma::vec allProbabilities1(N, arma::fill::ones);
  gDist2.Train(rdata, allProbabilities1);

  REQUIRE(gDist2.Alpha(0) == Approx(gDist.Alpha(0)).epsilon(1e-7));
  REQUIRE(gDist2.Beta(0) == Approx(gDist.Beta(0)).epsilon(1e-7));

  REQUIRE(gDist2.Alpha(1) == Approx(gDist.Alpha(1)).epsilon(1e-7));
  REQUIRE(gDist2.Beta(1) == Approx(gDist.Beta(1)).epsilon(1e-7));
}

/**
 * This test draws points from two different gamma distributions, sets the
 * probabilities for the points from the first distribution to something small
 * and the probabilities for the second to something large.  It ensures that the
 * gamma distribution recovered has the same parameters as the second gamma
 * distribution with high probabilities.
 */
TEST_CASE("GammaDistributionTrainTwoDistProbabilities1Test",
          "[DistributionTest]")
{
  double alphaReal = 5.4;
  double betaReal = 6.7;

  double alphaReal2 = 1.9;
  double betaReal2 = 8.4;

  // Create two gamma distribution random generators.
  std::gamma_distribution<double> dist(alphaReal, betaReal);
  std::gamma_distribution<double> dist2(alphaReal2, betaReal2);

  size_t N = 50000;
  size_t d = 2;
  arma::mat rdata(d, N);
  arma::vec probabilities(N);

  // Draw points alternately from the two different distributions.
  for (size_t j = 0; j < d; ++j)
  {
    for (size_t i = 0; i < N; ++i)
    {
      if (i % 2 == 0)
        rdata(j, i) = dist(math::randGen);
      else
        rdata(j, i) = dist2(math::randGen);
    }
  }

  for (size_t i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
      probabilities(i) = 0.02 * math::Random();
    else
      probabilities(i) = 0.98 + 0.02 * math::Random();
  }

  GammaDistribution gDist;
  gDist.Train(rdata, probabilities);

  REQUIRE(alphaReal2 == Approx(gDist.Alpha(0)).epsilon(0.05));
  REQUIRE(betaReal2 == Approx(gDist.Beta(0)).epsilon(0.05));

  REQUIRE(alphaReal2 == Approx(gDist.Alpha(1)).epsilon(0.05));
  REQUIRE(betaReal2 == Approx(gDist.Beta(1)).epsilon(0.05));
}

/**
 * This test verifies that the fitting procedure for GammaDistribution works
 * properly and converges near the actual gamma parameters. We do this twice
 * with different alpha/beta parameters so we make sure we don't have some weird
 * bug that always converges to the same number.
 */
TEST_CASE("GammaDistributionFittingTest", "[DistributionTest]")
{
  // Offset from the actual alpha/beta. 10% is quite a relaxed tolerance since
  // the random points we generate are few (for test speed) and might be fitted
  // better by a similar distribution.
  double errorTolerance = 10;

  size_t N = 5000;
  size_t d = 1; // Only 1 dimension is required for this.

  /** Iteration 1 (first parameter set) **/

  // Create a gamma-random generator and data.
  double alphaReal = 5.3;
  double betaReal = 1.5;
  std::gamma_distribution<double> dist(alphaReal, betaReal);

  // Random generation of gamma-like points.
  arma::mat rdata(d, N);
  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata(j, i) = dist(math::randGen);

  // Create Gamma object and call Train() on reference set.
  GammaDistribution gDist;
  gDist.Train(rdata);

  // Estimated parameter must be close to real.
  REQUIRE(gDist.Alpha(0) == Approx(alphaReal).epsilon(errorTolerance / 100));
  REQUIRE(gDist.Beta(0) == Approx(betaReal).epsilon(errorTolerance / 100));

  /** Iteration 2 (different parameter set) **/

  // Create a gamma-random generator and data.
  double alphaReal2 = 7.2;
  double betaReal2 = 0.9;
  std::gamma_distribution<double> dist2(alphaReal2, betaReal2);

  // Random generation of gamma-like points.
  arma::mat rdata2(d, N);
  for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i)
      rdata2(j, i) = dist2(math::randGen);

  // Create Gamma object and call Train() on reference set.
  GammaDistribution gDist2;
  gDist2.Train(rdata2);

  // Estimated parameter must be close to real.
  REQUIRE(gDist2.Alpha(0) == Approx(alphaReal2).epsilon(errorTolerance / 100));
  REQUIRE(gDist2.Beta(0) == Approx(betaReal2).epsilon(errorTolerance / 100));
}

/**
 * Test that Train() and the constructor that takes data give the same resulting
 * distribution.
 */
TEST_CASE("GammaDistributionTrainConstructorTest", "[DistributionTest]")
{
  const arma::mat data = arma::randu<arma::mat>(10, 500);

  GammaDistribution d1(data);
  GammaDistribution d2;
  d2.Train(data);

  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(d1.Alpha(i) == Approx(d2.Alpha(i)).epsilon(1e-7));
    REQUIRE(d1.Beta(i) == Approx(d2.Beta(i)).epsilon(1e-7));
  }
}

/**
 * Test that Train() with a dataset and Train() with dataset statistics return
 * the same results.
 */
TEST_CASE("GammaDistributionTrainStatisticsTest", "[DistributionTest]")
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

  REQUIRE(d1.Alpha(0) == Approx(d2.Alpha(0)).epsilon(1e-7));
  REQUIRE(d1.Beta(0) == Approx(d2.Beta(0)).epsilon(1e-7));
}

/**
 * Tests that Random() generates points that can be reasonably well fit by the
 * distribution that generated them.
 */
TEST_CASE("GammaDistributionRandomTest", "[DistributionTest]")
{
  const arma::vec a("2.0 2.5 3.0"), b("0.4 0.6 1.3");
  const size_t numPoints = 2000;

  // Distribution to generate points.
  GammaDistribution d1(a, b);
  arma::mat data(3, numPoints); // 3-d points.

  for (size_t i = 0; i < numPoints; ++i)
    data.col(i) = d1.Random();

  // Distribution to fit points.
  GammaDistribution d2(data);
  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE(d2.Alpha(i) == Approx(a(i)).epsilon(0.1)); // Within 10%
    REQUIRE(d2.Beta(i) == Approx(b(i)).epsilon(0.1));
  }
}

TEST_CASE("GammaDistributionProbabilityTest", "[DistributionTest]")
{
  // Train two 1-dimensional distributions.
  const arma::vec a1("2.0"), b1("0.9"), a2("3.1"), b2("1.4");
  arma::mat x1("2.0"), x2("2.94");
  arma::vec prob1, prob2;

  // Evaluated at wolfram|alpha
  GammaDistribution d1(a1, b1);
  d1.Probability(x1, prob1);
  REQUIRE(prob1(0) == Approx(0.267575).epsilon(1e-5));

  // Evaluated at wolfram|alpha
  GammaDistribution d2(a2, b2);
  d2.Probability(x2, prob2);
  REQUIRE(prob2(0) == Approx(0.189043).epsilon(1e-5));

  // Check that the overload that returns the probability for 1 dimension
  // agrees.
  REQUIRE(prob2(0) == Approx(d2.Probability(2.94, 0)).epsilon(1e-7));

  // Combine into one 2-dimensional distribution.
  const arma::vec a3("2.0 3.1"), b3("0.9 1.4");
  arma::mat x3(2, 2);
  x3 << 2.0 << 2.94 << arma::endr
     << 2.0 << 2.94;
  arma::vec prob3;

  // Expect that the 2-dimensional distribution returns the product of the
  // 1-dimensional distributions (evaluated at wolfram|alpha).
  GammaDistribution d3(a3, b3);
  d3.Probability(x3, prob3);
  REQUIRE(prob3(0) == Approx(0.04408).epsilon(1e-4));
  REQUIRE(prob3(1) == Approx(0.026165).epsilon(1e-4));
}

TEST_CASE("GammaDistributionLogProbabilityTest", "[DistributionTest]")
{
  // Train two 1-dimensional distributions.
  const arma::vec a1("2.0"), b1("0.9"), a2("3.1"), b2("1.4");
  arma::mat x1("2.0"), x2("2.94");
  arma::vec logprob1, logprob2;

  // Evaluated at wolfram|alpha
  GammaDistribution d1(a1, b1);
  d1.LogProbability(x1, logprob1);
  REQUIRE(logprob1(0) == Approx(std::log(0.267575)).epsilon(1e-5));

  // Evaluated at wolfram|alpha
  GammaDistribution d2(a2, b2);
  d2.LogProbability(x2, logprob2);
  REQUIRE(logprob2(0) == Approx(std::log(0.189043)).epsilon(1e-5));

  // Check that the overload that returns the log probability for
  // 1 dimension agrees.
  REQUIRE(logprob2(0) == Approx(d2.LogProbability(2.94, 0)).epsilon(1e-7));

  // Combine into one 2-dimensional distribution.
  const arma::vec a3("2.0 3.1"), b3("0.9 1.4");
  arma::mat x3(2, 2);
  x3
    << 2.0 << 2.94 << arma::endr
    << 2.0 << 2.94;
  arma::vec logprob3;

  // Expect that the 2-dimensional distribution returns the product of the
  // 1-dimensional distributions (evaluated at wolfram|alpha).
  GammaDistribution d3(a3, b3);
  d3.LogProbability(x3, logprob3);
  REQUIRE(logprob3(0) == Approx(std::log(0.04408)).epsilon(1e-5));
  REQUIRE(logprob3(1) == Approx(std::log(0.026165)).epsilon(1e-5));
}

/**
 * Discrete Distribution serialization test.
 */
TEST_CASE("DiscreteDistributionTest", "[DistributionTest]")
{
  // I assume that I am properly saving vectors, so, this should be
  // straightforward.
  arma::vec prob;
  prob.randu(12);
  std::vector<arma::vec> probVector = std::vector<arma::vec>(1, prob);
  DiscreteDistribution t(probVector);

  DiscreteDistribution xmlT, textT, binaryT;

  // Load and save with all serializers.
  SerializeObjectAll(t, xmlT, textT, binaryT);

  for (size_t i = 0; i < 12; ++i)
  {
    arma::vec obs(1);
    obs[0] = i;
    const double prob = t.Probability(obs);
    if (prob == 0.0)
    {
      REQUIRE(xmlT.Probability(obs) == Approx(0.0).margin(1e-8));
      REQUIRE(textT.Probability(obs) == Approx(0.0).margin(1e-8));
      REQUIRE(binaryT.Probability(obs) == Approx(0.0).margin(1e-8));
    }
    else
    {
      REQUIRE(prob == Approx(xmlT.Probability(obs)).epsilon(1e-10));
      REQUIRE(prob == Approx(textT.Probability(obs)).epsilon(1e-10));
      REQUIRE(prob == Approx(binaryT.Probability(obs)).epsilon(1e-10));
    }
  }
}

/**
 * Gaussian Distribution serialization test.
 */
TEST_CASE("GaussianDistributionTest", "[DistributionTest]")
{
  arma::vec mean(10);
  mean.randu();
  // Generate a covariance matrix.
  arma::mat cov;
  cov.randu(10, 10);
  cov = (cov * cov.t());

  GaussianDistribution g(mean, cov);
  GaussianDistribution xmlG, textG, binaryG;

  SerializeObjectAll(g, xmlG, textG, binaryG);

  REQUIRE(g.Dimensionality() == xmlG.Dimensionality());
  REQUIRE(g.Dimensionality() == textG.Dimensionality());
  REQUIRE(g.Dimensionality() == binaryG.Dimensionality());

  // First, check the means.
  CheckMatrices(g.Mean(), xmlG.Mean(), textG.Mean(), binaryG.Mean());

  // Now, check the covariance.
  CheckMatrices(g.Covariance(), xmlG.Covariance(), textG.Covariance(),
      binaryG.Covariance());

  // Lastly, run some observations through and make sure the probability is the
  // same.  This should test anything cached internally.
  arma::mat randomObs;
  randomObs.randu(10, 500);

  for (size_t i = 0; i < 500; ++i)
  {
    const double prob = g.Probability(randomObs.unsafe_col(i));

    if (prob == 0.0)
    {
      REQUIRE(xmlG.Probability(randomObs.unsafe_col(i)) ==
          Approx(0.0).margin(1e-8));
      REQUIRE(textG.Probability(randomObs.unsafe_col(i)) ==
          Approx(0.0).margin(1e-8));
      REQUIRE(binaryG.Probability(randomObs.unsafe_col(i)) ==
          Approx(0.0).margin(1e-8));
    }
    else
    {
      REQUIRE(prob ==
          Approx(xmlG.Probability(randomObs.unsafe_col(i))).epsilon(1e-10));
      REQUIRE(prob ==
          Approx(textG.Probability(randomObs.unsafe_col(i))).epsilon(1e-10));
      REQUIRE(prob ==
          Approx(binaryG.Probability(randomObs.unsafe_col(i))).epsilon(1e-10));
    }
  }
}

/**
 * Laplace Distribution serialization test.
 */
TEST_CASE("LaplaceDistributionTest", "[DistributionTest]")
{
  arma::vec mean(20);
  mean.randu();

  LaplaceDistribution l(mean, 2.5);
  LaplaceDistribution xmlL, textL, binaryL;

  SerializeObjectAll(l, xmlL, textL, binaryL);

  REQUIRE(l.Scale() == Approx(xmlL.Scale()).epsilon(1e-10));
  REQUIRE(l.Scale() == Approx(textL.Scale()).epsilon(1e-10));
  REQUIRE(l.Scale() == Approx(binaryL.Scale()).epsilon(1e-10));

  CheckMatrices(l.Mean(), xmlL.Mean(), textL.Mean(), binaryL.Mean());
}

/**
 * Laplace Distribution Probability Test.
 */
TEST_CASE("LaplaceDistributionProbabilityTest", "[DistributionTest]")
{
  LaplaceDistribution l(arma::vec("0.0"), 1.0);

  // Simple case.
  REQUIRE(l.Probability(arma::vec("0.0")) ==
      Approx(0.500000000000000).epsilon(1e-7));
  REQUIRE(l.Probability(arma::vec("1.0")) ==
      Approx(0.183939720585721).epsilon(1e-7));

  arma::mat points = "0.0 1.0;";

  arma::vec probabilities;

  l.Probability(points, probabilities);

  REQUIRE(probabilities.n_elem == 2);

  REQUIRE(probabilities(0) == Approx(0.500000000000000).epsilon(1e-7));
  REQUIRE(probabilities(1) == Approx(0.183939720585721).epsilon(1e-7));
}

/**
 * Laplace Distribution Log Probability Test.
 */
TEST_CASE("LaplaceDistributionLogProbabilityTest", "[DistributionTest]")
{
  LaplaceDistribution l(arma::vec("0.0"), 1.0);

  // Simple case.
  REQUIRE(l.LogProbability(arma::vec("0.0")) ==
      Approx(-0.693147180559945).epsilon(1e-7));
  REQUIRE(l.LogProbability(arma::vec("1.0")) ==
      Approx(-1.693147180559946).epsilon(1e-7));

  arma::mat points = "0.0 1.0;";

  arma::vec logProbabilities;

  l.LogProbability(points, logProbabilities);

  REQUIRE(logProbabilities.n_elem == 2);

  REQUIRE(logProbabilities(0) ==
      Approx(-0.693147180559945).epsilon(1e-7));

  REQUIRE(logProbabilities(1) ==
      Approx(-1.693147180559946).epsilon(1e-7));
}

/**
 * Mahalanobis Distance serialization test.
 */
TEST_CASE("MahalanobisDistanceTest", "[DistributionTest]")
{
  MahalanobisDistance<> d;
  d.Covariance().randu(50, 50);

  MahalanobisDistance<> xmlD, textD, binaryD;

  SerializeObjectAll(d, xmlD, textD, binaryD);

  // Check the covariance matrices.
  CheckMatrices(d.Covariance(),
                xmlD.Covariance(),
                textD.Covariance(),
                binaryD.Covariance());
}

/**
 * Regression distribution serialization test.
 */
TEST_CASE("RegressionDistributionTest", "[DistributionTest]")
{
  // Generate some random data.
  arma::mat data;
  data.randn(15, 800);
  arma::rowvec responses;
  responses.randn(800);

  RegressionDistribution rd(data, responses);
  RegressionDistribution xmlRd, textRd, binaryRd;

  // Okay, now save it and load it.
  SerializeObjectAll(rd, xmlRd, textRd, binaryRd);

  // Check the gaussian distribution.
  CheckMatrices(rd.Err().Mean(),
                xmlRd.Err().Mean(),
                textRd.Err().Mean(),
                binaryRd.Err().Mean());
  CheckMatrices(rd.Err().Covariance(),
                xmlRd.Err().Covariance(),
                textRd.Err().Covariance(),
                binaryRd.Err().Covariance());

  // Check the regression function.
  if (rd.Rf().Lambda() == 0.0)
  {
    REQUIRE(xmlRd.Rf().Lambda() == Approx(0.0).margin(1e-8));
    REQUIRE(textRd.Rf().Lambda() == Approx(0.0).margin(1e-8));
    REQUIRE(binaryRd.Rf().Lambda() == Approx(0.0).margin(1e-8));
  }
  else
  {
    REQUIRE(rd.Rf().Lambda() == Approx(xmlRd.Rf().Lambda()).epsilon(1e-10));
    REQUIRE(rd.Rf().Lambda() == Approx(textRd.Rf().Lambda()).epsilon(1e-10));
    REQUIRE(rd.Rf().Lambda() == Approx(binaryRd.Rf().Lambda()).epsilon(1e-10));
  }

  CheckMatrices(rd.Rf().Parameters(),
                xmlRd.Rf().Parameters(),
                textRd.Rf().Parameters(),
                binaryRd.Rf().Parameters());
}

/*****************************************************/
/** Diagonal Covariance Gaussian Distribution Tests **/
/*****************************************************/

/**
 * Make sure Diagonal Covariance Gaussian distributions are initialized
 * correctly.
 */
TEST_CASE("DiagonalGaussianDistributionEmptyConstructor", "[DistributionTest]")
{
  DiagonalGaussianDistribution d;

  REQUIRE(d.Mean().n_elem == 0);
  REQUIRE(d.Covariance().n_elem == 0);
}

/**
 * Make sure Diagonal Covariance Gaussian distributions are initialized to
 * the correct dimensionality.
 */
TEST_CASE("DiagonalGaussianDistributionDimensionalityConstructor",
          "[DistributionTest]")
{
  DiagonalGaussianDistribution d(4);

  REQUIRE(d.Mean().n_elem == 4);
  REQUIRE(d.Covariance().n_elem == 4);
}

/**
 * Make sure Diagonal Covariance Gaussian distributions are initialized
 * correctly when we give a mean and covariance.
 */
TEST_CASE("DiagonalGaussianDistributionConstructor", "[DistributionTest]")
{
  arma::vec mean = arma::randu<arma::vec>(3);
  arma::vec covariance = arma::randu<arma::vec>(3);

  DiagonalGaussianDistribution d(mean, covariance);

  // Make sure the mean and covariance is correct.
  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE(d.Mean()(i) == Approx(mean(i)).epsilon(1e-7));
    REQUIRE(d.Covariance()(i) == Approx(covariance(i)).epsilon(1e-7));
  }
}

/**
 * Make sure the probability of observations is correct.
 * The values were calculated using 'dmvnorm' in R.
 */
TEST_CASE("DiagonalGaussianDistributionProbabilityTest", "[DistributionTest]")
{
  arma::vec mean("2 5 3 4 1");
  arma::vec cov("3 1 5 3 2");

  DiagonalGaussianDistribution d(mean, cov);

  // Observations lists randomly selected.
  REQUIRE(d.LogProbability("3 5 2 7 8") ==
    Approx(-20.861264167855161).epsilon(1e-7));
  REQUIRE(d.LogProbability("7 8 4 0 5") ==
    Approx(-22.277930834521829).epsilon(1e-7));
  REQUIRE(d.LogProbability("6 8 7 7 5") ==
    Approx(-21.111264167855161).epsilon(1e-7));
  REQUIRE(d.LogProbability("2 9 5 6 3") ==
    Approx(-16.9112641678551621).epsilon(1e-7));
  REQUIRE(d.LogProbability("5 8 2 9 7") ==
    Approx(-26.111264167855161).epsilon(1e-7));
}

/**
 * Test DiagonalGaussianDistribution::Probability() in the univariate case.
 * The values were calculated using 'dmvnorm' in R.
 */
TEST_CASE("DiagonalGaussianUnivariateProbabilityTest", "[DistributionTest]")
{
  DiagonalGaussianDistribution d(arma::vec("0.0"), arma::vec("1.0"));

  // Mean: 0.0, Covariance: 1.0
  REQUIRE(d.Probability("0.0") == Approx(0.3989422804014327).epsilon(1e-7));
  REQUIRE(d.Probability("1.0") == Approx(0.24197072451914337).epsilon(1e-7));
  REQUIRE(d.Probability("-1.0") == Approx(0.24197072451914337).epsilon(1e-7));

  // Mean: 0.0, Covariance: 2.0
  d.Covariance("2.0");
  REQUIRE(d.Probability("0.0") == Approx(0.28209479177387814).epsilon(1e-7));
  REQUIRE(d.Probability("1.0") == Approx(0.21969564473386122).epsilon(1e-7));
  REQUIRE(d.Probability("-1.0") == Approx(0.21969564473386122).epsilon(1e-7));

  // Mean: 1.0, Covariance: 1.0
  d.Mean() = "1.0";
  d.Covariance("1.0");
  REQUIRE(d.Probability("0.0") == Approx(0.24197072451914337).epsilon(1e-7));
  REQUIRE(d.Probability("1.0") == Approx(0.3989422804014327).epsilon(1e-7));
  REQUIRE(d.Probability("-1.0") == Approx(0.053990966513188056).epsilon(1e-7));

  // Mean: 1.0, Covariance: 2.0
  d.Covariance("2.0");
  REQUIRE(d.Probability("0.0") == Approx(0.21969564473386122).epsilon(1e-7));
  REQUIRE(d.Probability("1.0") == Approx(0.28209479177387814).epsilon(1e-7));
  REQUIRE(d.Probability("-1.0") == Approx(0.10377687435514872).epsilon(1e-7));
}

/**
 * Test DiagonalGaussianDistribution::Probability() in the multivariate case.
 * The values were calculated using 'dmvnorm' in R.
 */
TEST_CASE("DiagonalGaussianMultivariateProbabilityTest", "[DistributionTest]")
{
  arma::vec mean("0 0");
  arma::vec cov("2 2");
  arma::vec obs("0 0");

  DiagonalGaussianDistribution d(mean, cov);

  REQUIRE(d.Probability(obs) == Approx(0.079577471545947673).epsilon(1e-7));

  obs = "1 1";
  REQUIRE(d.Probability(obs) == Approx(0.048266176315026957).epsilon(1e-7));

  d.Mean() = "1 3";
  REQUIRE(d.Probability(obs) == Approx(0.029274915762159581).epsilon(1e-7));
  REQUIRE(d.Probability(-obs) == Approx(0.00053618878559782773).epsilon(1e-7));

  // Higher dimensional case.
  d.Mean() = "1 3 6 2 7";
  d.Covariance("3 1 5 3 2");
  obs = "2 5 7 3 8";
  REQUIRE(d.Probability(obs) == Approx(7.2790083003378082e-05).epsilon(1e-7));
}

/**
 * Test the phi() function, for multiple points in the multivariate Gaussian
 * case. The values were calculated using 'dmvnorm' in R.
 */
TEST_CASE("DiagonalGaussianMultipointMultivariateProbabilityTest",
          "[DistributionTest]")
{
  arma::vec mean = "2 5 3 7 2";
  arma::vec cov("9 2 1 4 8");
  arma::mat points = "3 5 2 7 5 8;"
                     "2 6 8 3 4 6;"
                     "1 4 2 7 8 2;"
                     "6 8 4 7 9 2;"
                     "4 6 7 7 3 2";
  arma::vec phis;
  DiagonalGaussianDistribution d(mean, cov);
  d.LogProbability(points, phis);

  REQUIRE(phis.n_elem == 6);

  REQUIRE(phis(0) == Approx(-12.453302051926864).epsilon(1e-7));
  REQUIRE(phis(1) == Approx(-10.147746496371308).epsilon(1e-7));
  REQUIRE(phis(2) == Approx(-13.210246496371308).epsilon(1e-7));
  REQUIRE(phis(3) == Approx(-19.724135385260197).epsilon(1e-7));
  REQUIRE(phis(4) == Approx(-21.585246496371308).epsilon(1e-7));
  REQUIRE(phis(5) == Approx(-13.647746496371308).epsilon(1e-7));
}

/**
 * Make sure random observations follow the probability distribution correctly.
 */
TEST_CASE("DiagonalGaussianDistributionRandomTest", "[DistributionTest]")
{
  arma::vec mean("2.5 1.25");
  arma::vec cov("0.50 0.25");

  DiagonalGaussianDistribution d(mean, cov);

  arma::mat obs(2, 5000);

  for (size_t i = 0; i < 5000; ++i)
    obs.col(i) = d.Random();

  // Make sure that reflects the actual distribution.
  arma::vec obsMean = arma::mean(obs, 1);
  arma::mat obsCov = mlpack::math::ColumnCovariance(obs);

  // 10% tolerance because this can be noisy.
  REQUIRE(obsMean(0) == Approx(mean(0)).epsilon(0.1));
  REQUIRE(obsMean(1) == Approx(mean(1)).epsilon(0.1));

  REQUIRE(obsCov(0, 0) == Approx(cov(0)).epsilon(0.1));
  REQUIRE(obsCov(1, 1) == Approx(cov(1)).epsilon(0.1));
}

/**
 * Make sure that we can properly estimate from given observations.
 */
TEST_CASE("DiagonalGaussianDistributionTrainTest", "[DistributionTest]")
{
  arma::vec mean("2.5 1.5 8.2 3.1");
  arma::vec cov("1.2 3.1 8.3 4.3");

  // Generate the observations.
  arma::mat observations(4, 10000);

  for (size_t i = 0; i < 10000; ++i)
    observations.col(i) = (arma::sqrt(cov) % arma::randn<arma::vec>(4)) + mean;

  DiagonalGaussianDistribution d;

  // Calculate the actual mean and covariance of data using armadillo.
  arma::vec actualMean = arma::mean(observations, 1);
  arma::mat actualCov = mlpack::math::ColumnCovariance(observations);

  // Estimate the parameters.
  d.Train(observations);

  // Check that the estimated parameters are right.
  for (size_t i = 0; i < 4; ++i)
  {
    REQUIRE(d.Mean()(i) - actualMean(i) == Approx(0.0).margin(1e-5));
    REQUIRE(d.Covariance()(i) - actualCov(i, i) == Approx(0.0).margin(1e-5));
  }
}

/**
 * Make sure the unbiased estimator of the weighted sample works correctly.
 * The values were calculated using 'cov.wt' in R.
 */
TEST_CASE("DiagonalGaussianUnbiasedEstimatorTest", "[DistributionTest]")
{
  // Generate the observations.
  arma::mat observations("3 5 2 7;"
                         "2 6 8 3;"
                         "1 4 2 7;"
                         "6 8 4 7");

  arma::vec probs("0.3 0.4 0.1 0.2");

  DiagonalGaussianDistribution d;

  // Estimate the parameters.
  d.Train(observations, probs);

  REQUIRE(d.Mean()(0) == Approx(4.5).epsilon(1e-7));
  REQUIRE(d.Mean()(1) == Approx(4.4).epsilon(1e-7));
  REQUIRE(d.Mean()(2) == Approx(3.5).epsilon(1e-7));
  REQUIRE(d.Mean()(3) == Approx(6.8).epsilon(1e-7));

  REQUIRE(d.Covariance()(0) == Approx(3.78571428571428603).epsilon(1e-7));
  REQUIRE(d.Covariance()(1) == Approx(6.34285714285714253).epsilon(1e-7));
  REQUIRE(d.Covariance()(2) == Approx(6.64285714285714235).epsilon(1e-7));
  REQUIRE(d.Covariance()(3) == Approx(2.22857142857142865).epsilon(1e-7));
}

/**
 * Make sure that if all weights are the same, i.e. w_i / V1 = 1 / N, then
 * the weighted mean and covariance reduce to the unweighted sample mean and
 * covariance.
 */
TEST_CASE("DiagonalGaussianWeightedParametersReductionTest", "[DistributionTest]")
{
  arma::vec mean("2.5 1.5 8.2 3.1");
  arma::vec cov("1.2 3.1 8.3 4.3");

  // Generate the observations.
  arma::mat obs(4, 5);
  arma::vec probs("0.2 0.2 0.2 0.2 0.2");

  for (size_t i = 0; i < 5; ++i)
    obs.col(i) = (arma::sqrt(cov) % arma::randn<arma::vec>(4)) + mean;

  DiagonalGaussianDistribution d1;
  DiagonalGaussianDistribution d2;

  // Estimate the parameters.
  d1.Train(obs);
  d2.Train(obs, probs);

  // Check if these are equal.
  for (size_t i = 0; i < 4; ++i)
  {
    REQUIRE(d1.Mean()(i) == Approx(d2.Mean()(i)).epsilon(1e-7));
    REQUIRE(d1.Covariance()(i) == Approx(d2.Covariance()(i)).epsilon(1e-7));
  }
}
