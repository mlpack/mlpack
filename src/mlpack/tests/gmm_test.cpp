/**
 * @file gmm_test.cpp
 * @author Ryan Curtin
 *
 * Test for the Gaussian Mixture Model class.
 */
#include <mlpack/core.h>

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
  // Initialize random seed.
  srand(time(NULL));

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
    gmm.ExpectationMaximization(data);

    arma::vec actual_mean = arma::mean(data, 1);
    arma::mat actual_covar = ccov(data, 1 /* biased estimator */);

    // Check the model to see that it is correct.
    BOOST_REQUIRE_CLOSE((gmm.Means()[0])[0], actual_mean(0), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Means()[0])[1], actual_mean(1), 1e-5);

    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(0, 0), actual_covar(0, 0), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(0, 1), actual_covar(0, 1), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(1, 0), actual_covar(1, 0), 1e-5);
    BOOST_REQUIRE_CLOSE((gmm.Covariances()[0])(1, 1), actual_covar(1, 1), 1e-5);

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

    covars[i].randu(dims, dims);
    covars[i] *= 2;

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
  gmm.ExpectationMaximization(data);

  arma::uvec sort_ref = sort_index(weights);
  arma::uvec sort_try = sort_index(gmm.Weights());

  // Check the model to see that it is correct.
  for (size_t i = 0; i < gaussians; i++)
  {
    // Check the mean.
    for (size_t j = 0; j < dims; j++)
      BOOST_REQUIRE_CLOSE((gmm.Means()[sort_try[i]])[j],
          (means[sort_ref[i]])[j], 1e-5);

    // Check the covariance.
    for (size_t row = 0; row < dims; row++)
      for (size_t col = 0; col < dims; col++)
        BOOST_REQUIRE_CLOSE((gmm.Covariances()[sort_try[i]])(row, col),
            (covars[sort_ref[i]])(row, col), 1e-5);

    // Check the weight.
    BOOST_REQUIRE_CLOSE(gmm.Weights()[sort_try[i]], weights[sort_ref[i]],
        1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
