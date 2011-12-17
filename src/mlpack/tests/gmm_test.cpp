/**
 * @file gmm_test.cpp
 * @author Ryan Curtin
 *
 * Test for the Gaussian Mixture Model class.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/gmm/phi.hpp>

#include <boost/test/unit_test.hpp>

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
    data.randn(2 /* dimension */, 100 * pow(10, (iterations / 3.0)));

    // Now apply mean and covariance.
    data.row(0) *= covar(0);
    data.row(1) *= covar(1);

    data.row(0) += mean(0);
    data.row(1) += mean(1);

    // Now, train the model.
    GMM gmm(1, 2);
    gmm.Estimate(data);

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
  GMM gmm(gaussians, dims);
  gmm.Estimate(data);

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
  // Generate observations from a Gaussian distribution.
  distribution::GaussianDistribution d("0.5 1.0", "1.0 0.3; 0.3 1.0");

  // 10000 observations, each with random probability.
  arma::mat observations(2, 20000);
  for (size_t i = 0; i < 20000; i++)
    observations.col(i) = d.Random();
  arma::vec probabilities;
  probabilities.randu(20000); // Random probabilities.

  // Now train the model.
  GMM g(1, 2);

  g.Estimate(observations, probabilities);

  // Check that it is trained correctly.  5% tolerance because of random error
  // present in observations.
  BOOST_REQUIRE_CLOSE(g.Means()[0][0], 0.5, 5.0);
  BOOST_REQUIRE_CLOSE(g.Means()[0][1], 1.0, 5.0);

  // 7% tolerance on the large numbers, 10% on the smaller numbers.
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](0, 0), 1.0, 7.0);
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](0, 1), 0.3, 10.0);
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](1, 0), 0.3, 10.0);
  BOOST_REQUIRE_CLOSE(g.Covariances()[0](1, 1), 1.0, 7.0);

  BOOST_REQUIRE_CLOSE(g.Weights()[0], 1.0, 1e-5);
}

/**
 * Train a multi-Gaussian mixture, using the overload of Estimate() where
 * probabilities of the observation are given.
 */
BOOST_AUTO_TEST_CASE(GMMTrainEMMultipleGaussiansWithProbability)
{
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
                                                       "0.0 0.1 1.0");

  // Now we'll generate points and probabilities.  1500 points.  Slower than I
  // would like...
  arma::mat points(3, 1500);
  arma::vec probabilities(1500);

  for (size_t i = 0; i < 1500; i++)
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
  GMM g(3, 3); // 3 dimensions, 3 components.

  g.Estimate(points, probabilities);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(g.Weights());

  // The tolerances in our checks are quite large, but it is good to remember
  // that we introduced a fair amount of random noise into this whole process.

  // First Gaussian (g1).
  BOOST_REQUIRE_SMALL(g.Weights()[sortedIndices[0]] - 0.2222222222222, 0.075);

  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_SMALL((g.Means()[sortedIndices[0]][i] - d1.Mean()[i]), 0.25);

  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL((g.Covariances()[sortedIndices[0]](row, col) -
          d1.Covariance()(row, col)), 0.60); // Big tolerance!  Lots of noise.

  // Second Gaussian (g2).
  BOOST_REQUIRE_SMALL(g.Weights()[sortedIndices[1]] - 0.3333333333333, 0.075);

  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_SMALL((g.Means()[sortedIndices[1]][i] - d2.Mean()[i]), 0.25);

  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL((g.Covariances()[sortedIndices[1]](row, col) -
          d2.Covariance()(row, col)), 0.55); // Big tolerance!  Lots of noise.

  // Third Gaussian (g3).
  BOOST_REQUIRE_SMALL(g.Weights()[sortedIndices[2]] - 0.4444444444444, 0.1);

  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_SMALL((g.Means()[sortedIndices[2]][i] - d3.Mean()[i]), 0.25);

  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL((g.Covariances()[sortedIndices[2]](row, col) -
          d3.Covariance()(row, col)), 0.50); // Big tolerance!  Lots of noise.
}

/**
 * Make sure generating observations randomly works.  We'll do this by
 * generating a bunch of random observations and then re-training on them, and
 * hope that our model is the same.
 */
BOOST_AUTO_TEST_CASE(GMMRandomTest)
{
  // Simple GMM distribution.
  GMM gmm(2, 2);
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
  GMM gmm2(2, 2);
  gmm2.Estimate(observations);

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
BOOST_AUTO_TEST_SUITE_END();
