/**
 * @file gmm_test.cpp
 * @author Ryan Curtin
 *
 * Test for the Gaussian Mixture Model class.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/gmm/phi.hpp>

#include <mlpack/methods/gmm/no_constraint.hpp>
#include <mlpack/methods/gmm/positive_definite_constraint.hpp>
#include <mlpack/methods/gmm/diagonal_constraint.hpp>
#include <mlpack/methods/gmm/eigenvalue_ratio_constraint.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::gmm;

BOOST_AUTO_TEST_SUITE(GMMTest);
/**
 * Test the phi() function, in the univariate Gaussian case.
 */
BOOST_AUTO_TEST_CASE(UnivariatePhiTest)
{
  // Simple case.
  BOOST_REQUIRE_CLOSE(phi(0.0, 0.0, 1.0), 0.398942280401433, 1e-5);

  // A few more cases...
  BOOST_REQUIRE_CLOSE(phi(0.0, 0.0, 2.0), 0.282094791773878, 1e-5);

  BOOST_REQUIRE_CLOSE(phi(1.0, 0.0, 1.0), 0.241970724519143, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-1.0, 0.0, 1.0), 0.241970724519143, 1e-5);

  BOOST_REQUIRE_CLOSE(phi(1.0, 0.0, 2.0), 0.219695644733861, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-1.0, 0.0, 2.0), 0.219695644733861, 1e-5);

  BOOST_REQUIRE_CLOSE(phi(1.0, 1.0, 1.0), 0.398942280401433, 1e-5);

  BOOST_REQUIRE_CLOSE(phi(-1.0, 1.0, 2.0), 0.103776874355149, 1e-5);
}

/**
 * Test the phi() function, in the multivariate Gaussian case.
 */
BOOST_AUTO_TEST_CASE(MultivariatePhiTest)
{
  // Simple case.
  arma::vec mean = "0 0";
  arma::mat cov = "1 0; 0 1";
  arma::vec x = "0 0";

  BOOST_REQUIRE_CLOSE(phi(x, mean, cov), 0.159154943091895, 1e-5);

  cov = "2 0; 0 2";

  BOOST_REQUIRE_CLOSE(phi(x, mean, cov), 0.0795774715459477, 1e-5);

  x = "1 1";

  BOOST_REQUIRE_CLOSE(phi(x, mean, cov), 0.0482661763150270, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-x, mean, cov), 0.0482661763150270, 1e-5);

  mean = "1 1";

  BOOST_REQUIRE_CLOSE(phi(x, mean, cov), 0.0795774715459477, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-x, -mean, cov), 0.0795774715459477, 1e-5);

  cov = "2 1.5; 1 4";

  BOOST_REQUIRE_CLOSE(phi(x, mean, cov), 0.0624257046546403, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-x, -mean, cov), 0.0624257046546403, 1e-5);

  x = "-1 4";

  BOOST_REQUIRE_CLOSE(phi(x, mean, cov), 0.00144014867515135, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-x, mean, cov), 0.00133352162064845, 1e-5);

  // Higher-dimensional case.
  x = "0 1 2 3 4";
  mean = "5 6 3 3 2";
  cov = "6 1 1 0 2;"
        "0 7 1 0 1;"
        "1 1 4 1 1;"
        "1 0 1 7 0;"
        "2 0 1 1 6";

  BOOST_REQUIRE_CLOSE(phi(x, mean, cov), 1.02531207499358e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-x, -mean, cov), 1.02531207499358e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(x, -mean, cov), 1.06784794079363e-8, 1e-5);
  BOOST_REQUIRE_CLOSE(phi(-x, mean, cov), 1.06784794079363e-8, 1e-5);
}

/**
 * Test the phi() function, for multiple points in the multivariate Gaussian
 * case.
 */
BOOST_AUTO_TEST_CASE(MultipointMultivariatePhiTest)
{
  // Same case as before.
  arma::vec mean = "5 6 3 3 2";
  arma::mat cov = "6 1 1 0 2; 0 7 1 0 1; 1 1 4 1 1; 1 0 1 7 0; 2 0 1 1 6";

  arma::mat points = "0 3 2 2 3 4;"
                     "1 2 2 1 0 0;"
                     "2 3 0 5 5 6;"
                     "3 7 8 0 1 1;"
                     "4 8 1 1 0 0;";

  arma::vec phis;
  phi(points, mean, cov, phis);

  BOOST_REQUIRE_EQUAL(phis.n_elem, 6);

  BOOST_REQUIRE_CLOSE(phis(0), 1.02531207499358e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(1), 1.82353695848039e-7, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(2), 1.29759261892949e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(3), 1.33218060268258e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(4), 1.12120427975708e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(phis(5), 4.57951032485297e-7, 1e-5);
}

/**
 * Test GMM::Probability() for a single observation for a few cases.
 */
BOOST_AUTO_TEST_CASE(GMMProbabilityTest)
{
  // Create a GMM.
  GMM<> gmm(2, 2);
  gmm.Means()[0] = "0 0";
  gmm.Means()[1] = "3 3";
  gmm.Covariances()[0] = "1 0; 0 1";
  gmm.Covariances()[1] = "2 1; 1 2";
  gmm.Weights() = "0.3 0.7";

  // Now test a couple observations.  These comparisons are calculated by hand.
  BOOST_REQUIRE_CLOSE(gmm.Probability("0 0"), 0.05094887202, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("1 1"), 0.03451996667, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("2 2"), 0.04696302254, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("3 3"), 0.06432759685, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("-1 5.3"), 2.503171278804e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("1.4 0"), 0.024676682176, 1e-5);
}

/**
 * Test GMM::Probability() for a single observation being from a particular
 * component.
 */
BOOST_AUTO_TEST_CASE(GMMProbabilityComponentTest)
{
  // Create a GMM (same as the last test).
  GMM<> gmm(2, 2);
  gmm.Means()[0] = "0 0";
  gmm.Means()[1] = "3 3";
  gmm.Covariances()[0] = "1 0; 0 1";
  gmm.Covariances()[1] = "2 1; 1 2";
  gmm.Weights() = "0.3 0.7";

  // Now test a couple observations.  These comparisons are calculated by hand.
  BOOST_REQUIRE_CLOSE(gmm.Probability("0 0", 0), 0.0477464829276, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("0 0", 1), 0.0032023890978, 1e-5);

  BOOST_REQUIRE_CLOSE(gmm.Probability("1 1", 0), 0.0175649494573, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("1 1", 1), 0.0169550172159, 1e-5);

  BOOST_REQUIRE_CLOSE(gmm.Probability("2 2", 0), 8.7450733951e-4, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("2 2", 1), 0.0460885151993, 1e-5);

  BOOST_REQUIRE_CLOSE(gmm.Probability("3 3", 0), 5.8923841039e-6, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("3 3", 1), 0.0643217044658, 1e-5);

  BOOST_REQUIRE_CLOSE(gmm.Probability("-1 5.3", 0), 2.30212100302e-8, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("-1 5.3", 1), 2.48015006877e-6, 1e-5);

  BOOST_REQUIRE_CLOSE(gmm.Probability("1.4 0", 0), 0.0179197849738, 1e-5);
  BOOST_REQUIRE_CLOSE(gmm.Probability("1.4 0", 1), 0.0067568972024, 1e-5);
}

/**
 * Test training a model on only one Gaussian (randomly generated) in two
 * dimensions.  We will vary the dataset size from small to large.  The EM
 * algorithm is used for training the GMM.
 */
BOOST_AUTO_TEST_CASE(GMMTrainEMOneGaussian)
{
  for (size_t iterations = 0; iterations < 4; iterations++)
  {
    // Determine random covariance and mean.
    arma::vec mean;
    mean.randu(2);
    arma::vec covar;
    covar.randu(2);

    arma::mat data;
    data.randn(2 /* dimension */, 150 * pow(10, (iterations / 3.0)));

    // Now apply mean and covariance.
    data.row(0) *= covar(0);
    data.row(1) *= covar(1);

    data.row(0) += mean(0);
    data.row(1) += mean(1);

    // Now, train the model.
    GMM<> gmm(1, 2);
    gmm.Estimate(data, 10);

    arma::vec actualMean = arma::mean(data, 1);
    arma::mat actualCovar = ccov(data, 1 /* biased estimator */);

    // Check the model to see that it is correct.
    BOOST_REQUIRE_CLOSE((gmm.Means()[0])[0], actualMean(0), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Means()[0])[1], actualMean(1), 1e-5);

    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(0, 0), actualCovar(0, 0), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(0, 1), actualCovar(0, 1), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(1, 0), actualCovar(1, 0), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(1, 1), actualCovar(1, 1), 1e-5);

    BOOST_REQUIRE_CLOSE(gmm.Weights()[0], 1.0, 1e-5);
  }
}

/**
 * Test a training model on multiple Gaussians in higher dimensionality than
 * two.  We will hold the dataset size constant at 10k points.  The EM algorithm
 * is used for training the GMM.
 */
BOOST_AUTO_TEST_CASE(GMMTrainEMMultipleGaussians)
{
  // Higher dimensionality gives us a greater chance of having separated
  // Gaussians.
  size_t dims = 8;
  size_t gaussians = 3;

  // Generate dataset.
  arma::mat data;
  data.zeros(dims, 500);

  std::vector<arma::vec> means(gaussians);
  std::vector<arma::mat> covars(gaussians);
  arma::vec weights(gaussians);
  arma::Col<size_t> counts(gaussians);

  // Choose weights randomly.
  weights.zeros();
  while (weights.min() < 0.02)
  {
    weights.randu(gaussians);
    weights /= accu(weights);
  }

  for (size_t i = 0; i < gaussians; i++)
    counts[i] = round(weights[i] * (data.n_cols - gaussians));
  // Ensure one point minimum in each.
  counts += 1;

  // Account for rounding errors (possibly necessary).
  counts[gaussians - 1] += (data.n_cols - arma::accu(counts));

  // Build each Gaussian individually.
  size_t point = 0;
  for (size_t i = 0; i < gaussians; i++)
  {
    arma::mat gaussian;
    gaussian.randn(dims, counts[i]);

    // Randomly generate mean and covariance.
    means[i].randu(dims);
    means[i] -= 0.5;
    means[i] *= 50;

    // We need to make sure the covariance is positive definite.  We will take a
    // random matrix C and then set our covariance to 4 * C * C', which will be
    // positive semidefinite.
    covars[i].randu(dims, dims);
    covars[i] *= 4 * trans(covars[i]);

    data.cols(point, point + counts[i] - 1) = (covars[i] * gaussian + means[i]
        * arma::ones<arma::rowvec>(counts[i]));

    // Calculate the actual means and covariances because they will probably
    // be different (this is easier to do before we shuffle the points).
    means[i] = arma::mean(data.cols(point, point + counts[i] - 1), 1);
    covars[i] = ccov(data.cols(point, point + counts[i] - 1), 1 /* biased */);

    point += counts[i];
  }

  // Calculate actual weights.
  for (size_t i = 0; i < gaussians; i++)
    weights[i] = (double) counts[i] / data.n_cols;

  // Now train the model.
  GMM<> gmm(gaussians, dims);
  gmm.Estimate(data, 10);

  arma::uvec sortRef = sort_index(weights);
  arma::uvec sortTry = sort_index(gmm.Weights());

  // Check the model to see that it is correct.
  for (size_t i = 0; i < gaussians; i++)
  {
    // Check the mean.
    for (size_t j = 0; j < dims; j++)
      BOOST_REQUIRE_CLOSE((gmm.Means()[sortTry[i]])[j],
          (means[sortRef[i]])[j], 1e-5);

    // Check the covariance.
    for (size_t row = 0; row < dims; row++)
      for (size_t col = 0; col < dims; col++)
        BOOST_REQUIRE_CLOSE((gmm.Covariances()[sortTry[i]])(row, col),
            (covars[sortRef[i]])(row, col), 1e-5);

    // Check the weight.
    BOOST_REQUIRE_CLOSE(gmm.Weights()[sortTry[i]], weights[sortRef[i]],
        1e-5);
  }
}

/**
 * Train a single-gaussian mixture, but using the overload of Estimate() where
 * probabilities of the observation are given.
 */
BOOST_AUTO_TEST_CASE(GMMTrainEMSingleGaussianWithProbability)
{
  math::RandomSeed(std::time(NULL));

  // Generate observations from a Gaussian distribution.
  distribution::GaussianDistribution d("0.5 1.0", "1.0 0.3; 0.3 1.0");

  // 10000 observations, each with random probability.
  arma::mat observations(2, 20000);
  for (size_t i = 0; i < 20000; i++)
    observations.col(i) = d.Random();
  arma::vec probabilities;
  probabilities.randu(20000); // Random probabilities.

  // Now train the model.
  GMM<> g(1, 2);
  g.Estimate(observations, probabilities, 10);

  // Check that it is trained correctly.  5% tolerance because of random error
  // present in observations.
  BOOST_REQUIRE_CLOSE(g.Means()[0][0], 0.5, 5.0);
  BOOST_REQUIRE_CLOSE(g.Means()[0][1], 1.0, 5.0);

  // 6% tolerance on the large numbers, 10% on the smaller numbers.
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](0, 0), 1.0, 6.0);
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](0, 1), 0.3, 10.0);
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](1, 0), 0.3, 10.0);
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](1, 1), 1.0, 6.0);

  BOOST_REQUIRE_CLOSE(g.Weights()[0], 1.0, 1e-5);
}

/**
 * Train a multi-Gaussian mixture, using the overload of Estimate() where
 * probabilities of the observation are given.
 */
BOOST_AUTO_TEST_CASE(GMMTrainEMMultipleGaussiansWithProbability)
{
  srand(time(NULL));

  // We'll have three Gaussian distributions from this mixture, and one Gaussian
  // not from this mixture (but we'll put some observations from it in).
  distribution::GaussianDistribution d1("0.0 1.0 0.0", "1.0 0.0 0.5;"
                                                       "0.0 0.8 0.1;"
                                                       "0.5 0.1 1.0");
  distribution::GaussianDistribution d2("2.0 -1.0 5.0", "3.0 0.0 0.5;"
                                                        "0.0 1.2 0.2;"
                                                        "0.5 0.2 1.3");
  distribution::GaussianDistribution d3("0.0 5.0 -3.0", "2.0 0.0 0.0;"
                                                        "0.0 0.3 0.0;"
                                                        "0.0 0.0 1.0");
  distribution::GaussianDistribution d4("4.0 2.0 2.0", "1.5 0.6 0.5;"
                                                       "0.6 1.1 0.1;"
                                                       "0.5 0.1 1.0");

  // Now we'll generate points and probabilities.  1500 points.  Slower than I
  // would like...
  arma::mat points(3, 2000);
  arma::vec probabilities(2000);

  for (size_t i = 0; i < 2000; i++)
  {
    double randValue = math::Random();

    if (randValue <= 0.20) // p(d1) = 0.20
      points.col(i) = d1.Random();
    else if (randValue <= 0.50) // p(d2) = 0.30
      points.col(i) = d2.Random();
    else if (randValue <= 0.90) // p(d3) = 0.40
      points.col(i) = d3.Random();
    else // p(d4) = 0.10
      points.col(i) = d4.Random();

    // Set the probability right.  If it came from this mixture, it should be
    // 0.97 plus or minus a little bit of noise.  If not, then it should be 0.03
    // plus or minus a little bit of noise.  The base probability (minus the
    // noise) is parameterizable for easy modification of the test.
    double confidence = 0.995;
    double perturbation = math::Random(-0.005, 0.005);

    if (randValue <= 0.90)
      probabilities(i) = confidence + perturbation;
    else
      probabilities(i) = (1 - confidence) + perturbation;
  }

  // Now train the model.
  GMM<> g(4, 3); // 3 dimensions, 4 components.

  g.Estimate(points, probabilities, 8);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(g.Weights());

  // The tolerances in our checks are quite large, but it is good to remember
  // that we introduced a fair amount of random noise into this whole process.

  // First Gaussian (d4).
  BOOST_REQUIRE_SMALL(g.Weights()[sortedIndices[0]] - 0.1, 0.1);

  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_SMALL((g.Means()[sortedIndices[0]][i] - d4.Mean()[i]), 0.4);

  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL((g.Covariances()[sortedIndices[0]](row, col) -
          d4.Covariance()(row, col)), 0.60); // Big tolerance!  Lots of noise.

  // Second Gaussian (d1).
  BOOST_REQUIRE_SMALL(g.Weights()[sortedIndices[1]] - 0.2, 0.1);

  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_SMALL((g.Means()[sortedIndices[1]][i] - d1.Mean()[i]), 0.4);

  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL((g.Covariances()[sortedIndices[1]](row, col) -
          d1.Covariance()(row, col)), 0.55); // Big tolerance!  Lots of noise.

  // Third Gaussian (d2).
  BOOST_REQUIRE_SMALL(g.Weights()[sortedIndices[2]] - 0.3, 0.1);

  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_SMALL((g.Means()[sortedIndices[2]][i] - d2.Mean()[i]), 0.4);

  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL((g.Covariances()[sortedIndices[2]](row, col) -
          d2.Covariance()(row, col)), 0.50); // Big tolerance!  Lots of noise.

  // Fourth gaussian (d3).
  BOOST_REQUIRE_SMALL(g.Weights()[sortedIndices[3]] - 0.4, 0.1);

  for (size_t i = 0; i < 3; ++i)
    BOOST_REQUIRE_SMALL((g.Means()[sortedIndices[3]][i] - d3.Mean()[i]), 0.4);

  for (size_t row = 0; row < 3; ++row)
    for (size_t col = 0; col < 3; ++col)
      BOOST_REQUIRE_SMALL((g.Covariances()[sortedIndices[3]](row, col) -
          d3.Covariance()(row, col)), 0.50);
}

/**
 * Make sure generating observations randomly works.  We'll do this by
 * generating a bunch of random observations and then re-training on them, and
 * hope that our model is the same.
 */
BOOST_AUTO_TEST_CASE(GMMRandomTest)
{
  // Simple GMM distribution.
  GMM<> gmm(2, 2);
  gmm.Weights() = arma::vec("0.40 0.60");

  // N([2.25 3.10], [1.00 0.20; 0.20 0.89])
  gmm.Means()[0] = arma::vec("2.25 3.10");
  gmm.Covariances()[0] = arma::mat("1.00 0.60; 0.60 0.89");

  // N([4.10 1.01], [1.00 0.00; 0.00 1.01])
  gmm.Means()[1] = arma::vec("4.10 1.01");
  gmm.Covariances()[1] = arma::mat("1.00 0.70; 0.70 1.01");

  // Now generate a bunch of observations.
  arma::mat observations(2, 4000);
  for (size_t i = 0; i < 4000; i++)
    observations.col(i) = gmm.Random();

  // A new one which we'll train.
  GMM<> gmm2(2, 2);
  gmm2.Estimate(observations, 10);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(gmm2.Weights());

  // Now check that the parameters are the same.  Tolerances are kind of big
  // because we only used 2000 observations.
  BOOST_REQUIRE_CLOSE(gmm.Weights()[0], gmm2.Weights()[sortedIndices[0]], 7.0);
  BOOST_REQUIRE_CLOSE(gmm.Weights()[1], gmm2.Weights()[sortedIndices[1]], 7.0);

  BOOST_REQUIRE_CLOSE(gmm.Means()[0][0], gmm2.Means()[sortedIndices[0]][0],
      6.5);
  BOOST_REQUIRE_CLOSE(gmm.Means()[0][1], gmm2.Means()[sortedIndices[0]][1],
      6.5);

  BOOST_REQUIRE_CLOSE(gmm.Covariances()[0](0, 0),
      gmm2.Covariances()[sortedIndices[0]](0, 0), 13.0);
  BOOST_REQUIRE_CLOSE(gmm.Covariances()[0](0, 1),
      gmm2.Covariances()[sortedIndices[0]](0, 1), 22.0);
  BOOST_REQUIRE_CLOSE(gmm.Covariances()[0](1, 0),
      gmm2.Covariances()[sortedIndices[0]](1, 0), 22.0);
  BOOST_REQUIRE_CLOSE(gmm.Covariances()[0](1, 1),
      gmm2.Covariances()[sortedIndices[0]](1, 1), 13.0);

  BOOST_REQUIRE_CLOSE(gmm.Means()[1][0], gmm2.Means()[sortedIndices[1]][0],
      6.5);
  BOOST_REQUIRE_CLOSE(gmm.Means()[1][1], gmm2.Means()[sortedIndices[1]][1],
      6.5);

  BOOST_REQUIRE_CLOSE(gmm.Covariances()[1](0, 0),
      gmm2.Covariances()[sortedIndices[1]](0, 0), 13.0);
  BOOST_REQUIRE_CLOSE(gmm.Covariances()[1](0, 1),
      gmm2.Covariances()[sortedIndices[1]](0, 1), 22.0);
  BOOST_REQUIRE_CLOSE(gmm.Covariances()[1](1, 0),
      gmm2.Covariances()[sortedIndices[1]](1, 0), 22.0);
  BOOST_REQUIRE_CLOSE(gmm.Covariances()[1](1, 1),
      gmm2.Covariances()[sortedIndices[1]](1, 1), 13.0);
}

/**
 * Test classification of observations by component.
 */
BOOST_AUTO_TEST_CASE(GMMClassifyTest)
{
  // First create a Gaussian with a few components.
  GMM<> gmm(3, 2);
  gmm.Means()[0] = "0 0";
  gmm.Means()[1] = "1 3";
  gmm.Means()[2] = "-2 -2";
  gmm.Covariances()[0] = "1 0; 0 1";
  gmm.Covariances()[1] = "3 2; 2 3";
  gmm.Covariances()[2] = "2.2 1.4; 1.4 5.1";
  gmm.Weights() = "0.6 0.25 0.15";

  arma::mat observations = arma::trans(arma::mat(
    " 0  0;"
    " 0  1;"
    " 0  2;"
    " 1 -2;"
    " 2 -2;"
    "-2  0;"
    " 5  5;"
    "-2 -2;"
    " 3  3;"
    "25 25;"
    "-1 -1;"
    "-3 -3;"
    "-5  1"));

  arma::Col<size_t> classes;

  gmm.Classify(observations, classes);

  // Test classification of points.  Classifications produced by hand.
  BOOST_REQUIRE_EQUAL(classes[ 0], 0);
  BOOST_REQUIRE_EQUAL(classes[ 1], 0);
  BOOST_REQUIRE_EQUAL(classes[ 2], 1);
  BOOST_REQUIRE_EQUAL(classes[ 3], 0);
  BOOST_REQUIRE_EQUAL(classes[ 4], 0);
  BOOST_REQUIRE_EQUAL(classes[ 5], 0);
  BOOST_REQUIRE_EQUAL(classes[ 6], 1);
  BOOST_REQUIRE_EQUAL(classes[ 7], 2);
  BOOST_REQUIRE_EQUAL(classes[ 8], 1);
  BOOST_REQUIRE_EQUAL(classes[ 9], 1);
  BOOST_REQUIRE_EQUAL(classes[10], 0);
  BOOST_REQUIRE_EQUAL(classes[11], 2);
  BOOST_REQUIRE_EQUAL(classes[12], 2);
}

BOOST_AUTO_TEST_CASE(GMMLoadSaveTest)
{
  // Create a GMM, save it, and load it.
  GMM<> gmm(10, 4);
  gmm.Weights().randu();

  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    gmm.Means()[i].randu();
    gmm.Covariances()[i].randu();
  }

  gmm.Save("test-gmm-save.xml");

  GMM<> gmm2;
  gmm2.Load("test-gmm-save.xml");

  // Remove clutter.
  remove("test-gmm-save.xml");

  BOOST_REQUIRE_EQUAL(gmm.Gaussians(), gmm2.Gaussians());
  BOOST_REQUIRE_EQUAL(gmm.Dimensionality(), gmm2.Dimensionality());

  for (size_t i = 0; i < gmm.Dimensionality(); ++i)
    BOOST_REQUIRE_CLOSE(gmm.Weights()[i], gmm2.Weights()[i], 1e-3);

  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
      BOOST_REQUIRE_CLOSE(gmm.Means()[i][j], gmm2.Means()[i][j], 1e-3);

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
      {
        BOOST_REQUIRE_CLOSE(gmm.Covariances()[i](j, k),
            gmm2.Covariances()[i](j, k), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(NoConstraintTest)
{
  // Generate random matrices and make sure they end up the same.
  for (size_t i = 0; i < 30; ++i)
  {
    const size_t rows = 5 + math::RandInt(100);
    const size_t cols = 5 + math::RandInt(100);
    arma::mat cov(rows, cols);
    cov.randu();
    arma::mat newcov(cov);

    NoConstraint::ApplyConstraint(newcov);

    for (size_t j = 0; j < cov.n_elem; ++j)
      BOOST_REQUIRE_CLOSE(newcov(j), cov(j), 1e-20);
  }
}

BOOST_AUTO_TEST_CASE(PositiveDefiniteConstraintTest)
{
  // Make sure matrices are made to be positive definite.
  for (size_t i = 0; i < 30; ++i)
  {
    const size_t elem = 5 + math::RandInt(50);
    arma::mat cov(elem, elem);
    cov.randu();

    PositiveDefiniteConstraint::ApplyConstraint(cov);

    BOOST_REQUIRE_GE((double) det(cov), 1e-50);
  }
}

BOOST_AUTO_TEST_CASE(DiagonalConstraintTest)
{
  // Make sure matrices are made to be positive definite.
  for (size_t i = 0; i < 30; ++i)
  {
    const size_t elem = 5 + math::RandInt(50);
    arma::mat cov(elem, elem);
    cov.randu();

    DiagonalConstraint::ApplyConstraint(cov);

    for (size_t j = 0; j < elem; ++j)
      for (size_t k = 0; k < elem; ++k)
        if (j != k)
          BOOST_REQUIRE_SMALL(cov(j, k), 1e-50);
  }
}

BOOST_AUTO_TEST_CASE(EigenvalueRatioConstraintTest)
{
  // Generate a list of eigenvalue ratios.
  arma::vec ratios("1.0 0.7 0.4 0.2 0.1 0.1 0.05 0.01");
  EigenvalueRatioConstraint erc(ratios);

  // Now make some random matrices and see if the constraint works.
  for (size_t i = 0; i < 30; ++i)
  {
    arma::mat cov(8, 8);
    cov.randu();

    erc.ApplyConstraint(cov);

    // Decompose the matrix and make sure things are right.
    arma::vec eigenvalues = arma::eig_sym(cov);

    for (size_t i = 0; i < eigenvalues.n_elem; ++i)
      BOOST_REQUIRE_CLOSE(eigenvalues[i] / eigenvalues[0], ratios[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(UseExistingModelTest)
{
  // If we run a GMM and it converges, then if we run it again using the
  // converged results as the starting point, then it should terminate after one
  // iteration and give basically the same results.

  // Higher dimensionality gives us a greater chance of having separated
  // Gaussians.
  size_t dims = 8;
  size_t gaussians = 3;

  // Generate dataset.
  arma::mat data;
  data.zeros(dims, 500);

  std::vector<arma::vec> means(gaussians);
  std::vector<arma::mat> covars(gaussians);
  arma::vec weights(gaussians);
  arma::Col<size_t> counts(gaussians);

  // Choose weights randomly.
  weights.zeros();
  while (weights.min() < 0.02)
  {
    weights.randu(gaussians);
    weights /= accu(weights);
  }

  for (size_t i = 0; i < gaussians; i++)
    counts[i] = round(weights[i] * (data.n_cols - gaussians));
  // Ensure one point minimum in each.
  counts += 1;

  // Account for rounding errors (possibly necessary).
  counts[gaussians - 1] += (data.n_cols - arma::accu(counts));

  // Build each Gaussian individually.
  size_t point = 0;
  for (size_t i = 0; i < gaussians; i++)
  {
    arma::mat gaussian;
    gaussian.randn(dims, counts[i]);

    // Randomly generate mean and covariance.
    means[i].randu(dims);
    means[i] -= 0.5;
    means[i] *= 50;

    // We need to make sure the covariance is positive definite.  We will take a
    // random matrix C and then set our covariance to 4 * C * C', which will be
    // positive semidefinite.
    covars[i].randu(dims, dims);
    covars[i] *= 4 * trans(covars[i]);

    data.cols(point, point + counts[i] - 1) = (covars[i] * gaussian + means[i]
        * arma::ones<arma::rowvec>(counts[i]));

    // Calculate the actual means and covariances because they will probably
    // be different (this is easier to do before we shuffle the points).
    means[i] = arma::mean(data.cols(point, point + counts[i] - 1), 1);
    covars[i] = ccov(data.cols(point, point + counts[i] - 1), 1 /* biased */);

    point += counts[i];
  }

  // Calculate actual weights.
  for (size_t i = 0; i < gaussians; i++)
    weights[i] = (double) counts[i] / data.n_cols;

  // Now train the model.
  GMM<> gmm(gaussians, dims);
  gmm.Estimate(data, 10);

  GMM<> oldgmm(gmm);

  // Retrain the model with the existing model as the starting point.
  gmm.Estimate(data, 1, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    BOOST_REQUIRE_CLOSE(gmm.Weights()[i], oldgmm.Weights()[i], 1e-4);

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      BOOST_REQUIRE_CLOSE(gmm.Means()[i][j], oldgmm.Means()[i][j], 1e-3);

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
        BOOST_REQUIRE_CLOSE(gmm.Covariances()[i](j, k),
                            oldgmm.Covariances()[i](j, k), 1e-3);
    }
  }

  // Do it again, with a larger number of trials.
  gmm = oldgmm;

  // Retrain the model with the existing model as the starting point.
  gmm.Estimate(data, 10, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    BOOST_REQUIRE_CLOSE(gmm.Weights()[i], oldgmm.Weights()[i], 1e-4);

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      BOOST_REQUIRE_CLOSE(gmm.Means()[i][j], oldgmm.Means()[i][j], 1e-3);

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
        BOOST_REQUIRE_CLOSE(gmm.Covariances()[i](j, k),
                            oldgmm.Covariances()[i](j, k), 1e-3);
    }
  }

  // Do it again, but using the overload of Estimate() that takes probabilities
  // into account.
  arma::vec probabilities(data.n_cols);
  probabilities.ones(); // Fill with ones.

  gmm = oldgmm;
  gmm.Estimate(data, probabilities, 1, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    BOOST_REQUIRE_CLOSE(gmm.Weights()[i], oldgmm.Weights()[i], 1e-4);

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      BOOST_REQUIRE_CLOSE(gmm.Means()[i][j], oldgmm.Means()[i][j], 1e-3);

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
        BOOST_REQUIRE_CLOSE(gmm.Covariances()[i](j, k),
                            oldgmm.Covariances()[i](j, k), 1e-3);
    }
  }

  // One more time, with multiple trials.
  gmm = oldgmm;
  gmm.Estimate(data, probabilities, 10, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    BOOST_REQUIRE_CLOSE(gmm.Weights()[i], oldgmm.Weights()[i], 1e-4);

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      BOOST_REQUIRE_CLOSE(gmm.Means()[i][j], oldgmm.Means()[i][j], 1e-3);

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
        BOOST_REQUIRE_CLOSE(gmm.Covariances()[i](j, k),
                            oldgmm.Covariances()[i](j, k), 1e-3);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
