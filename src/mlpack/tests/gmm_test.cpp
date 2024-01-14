/**
 * @file tests/gmm_test.cpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Test for the Gaussian Mixture Model class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/gmm.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;

/**
 * Test GMM::Probability() with a single observation at a time for a few cases.
 */

TEST_CASE("GMMProbabilityTest", "[GMMTest]")
{
  // Create a GMM.
  GMM gmm(2, 2);
  gmm.Component(0) = GaussianDistribution("0 0", "1 0; 0 1");
  gmm.Component(1) = GaussianDistribution("3 3", "2 1; 1 2");
  gmm.Weights() = "0.3 0.7";

  // Now test a couple observations.  These comparisons are calculated by hand.
  REQUIRE(gmm.Probability("0 0") == Approx(0.05094887202).epsilon(1e-7));
  REQUIRE(gmm.Probability("1 1") == Approx(0.03451996667).epsilon(1e-7));
  REQUIRE(gmm.Probability("2 2") == Approx(0.04696302254).epsilon(1e-7));
  REQUIRE(gmm.Probability("3 3") == Approx(0.06432759685).epsilon(1e-7));
  REQUIRE(gmm.Probability("-1 5.3") == Approx(2.503171278804e-6).epsilon(1e-7));
  REQUIRE(gmm.Probability("1.4 0") == Approx(0.024676682176).epsilon(1e-7));
}

/**
 * Test GMM::Probability() for a single observation being from a particular
 * component.
 */
TEST_CASE("GMMProbabilityComponentTest", "[GMMTest]")
{
  // Create a GMM (same as the last test).
  GMM gmm(2, 2);
  gmm.Component(0) = GaussianDistribution("0 0", "1 0; 0 1");
  gmm.Component(1) = GaussianDistribution("3 3", "2 1; 1 2");
  gmm.Weights() = "0.3 0.7";

  // Now test a couple observations.  These comparisons are calculated by hand.
  REQUIRE(gmm.Probability("0 0", 0) == Approx(0.0477464829276).epsilon(1e-7));
  REQUIRE(gmm.Probability("0 0", 1) == Approx(0.0032023890978).epsilon(1e-7));

  REQUIRE(gmm.Probability("1 1", 0) == Approx(0.0175649494573).epsilon(1e-7));
  REQUIRE(gmm.Probability("1 1", 1) == Approx(0.0169550172159).epsilon(1e-7));

  REQUIRE(gmm.Probability("2 2", 0) == Approx(8.7450733951e-4).epsilon(1e-7));
  REQUIRE(gmm.Probability("2 2", 1) == Approx(0.0460885151993).epsilon(1e-7));

  REQUIRE(gmm.Probability("3 3", 0) == Approx(5.8923841039e-6).epsilon(1e-7));
  REQUIRE(gmm.Probability("3 3", 1) == Approx(0.0643217044658).epsilon(1e-7));

  REQUIRE(gmm.Probability("-1 5.3", 0) ==
      Approx(2.30212100302e-8).epsilon(1e-7));
  REQUIRE(gmm.Probability("-1 5.3", 1) ==
      Approx(2.48015006877e-6).epsilon(1e-7));

  REQUIRE(gmm.Probability("1.4 0", 0) == Approx(0.0179197849738).epsilon(1e-7));
  REQUIRE(gmm.Probability("1.4 0", 1) == Approx(0.0067568972024).epsilon(1e-7));
}

/**
 * Test training a model on only one Gaussian (randomly generated) in two
 * dimensions.  We will vary the dataset size from small to large.  The EM
 * algorithm is used for training the GMM.
 */
TEST_CASE("GMMTrainEMOneGaussian", "[GMMTest]")
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
    GMM gmm(1, 2);
    gmm.Train(data, 10);

    arma::vec actualMean = arma::mean(data, 1);
    arma::mat actualCovar = ColumnCovariance(data,
        1 /* biased estimator */);

    // Check the model to see that it is correct.
    REQUIRE(arma::norm(gmm.Component(0).Mean() - actualMean) < 1e-5);
    REQUIRE(arma::norm(gmm.Component(0).Covariance() - actualCovar) < 1e-4);

    REQUIRE(gmm.Weights()[0] == Approx(1.0).epsilon(1e-7));
  }
}

/**
 * Test a training model on multiple Gaussians in higher dimensionality than
 * two.  We will hold the dataset size constant at 10k points.  The EM algorithm
 * is used for training the GMM.
 */
TEST_CASE("GMMTrainEMMultipleGaussians", "[GMMTest]")
{
  // Higher dimensionality gives us a greater chance of having separated
  // Gaussians.
  size_t dims = 8;
  size_t gaussians = 3;

  // We'll run three trials, and it needs to pass during at least one trial.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    // Generate dataset.
    arma::mat data;
    data.zeros(dims, 500);

    std::vector<arma::vec> means(gaussians);
    std::vector<arma::mat> covars(gaussians);
    arma::vec weights(gaussians);
    arma::Col<size_t> counts(gaussians);

    // Choose weights randomly.  We want each component to have somewhat
    // significant weight, but we also need to make sure that no weights are too
    // close.
    double minDiff = DBL_MAX;
    do
    {
      weights.zeros();
      weights.randu(gaussians);
      weights /= accu(weights);
      weights *= 0.4;
      weights += (0.6 / double(gaussians));
      weights /= accu(weights); // Paranoia, just to be sure they sum to 1.

      // Compute minimum element difference.
      minDiff = DBL_MAX;
      for (size_t i = 0; i < weights.n_elem; ++i)
        for (size_t j = (i + 1); j < weights.n_elem; ++j)
          if (std::abs(weights[i] - weights[j]) < minDiff)
            minDiff = std::abs(weights[i] - weights[j]);
    } while (minDiff < 0.02);

    for (size_t i = 0; i < gaussians; ++i)
      counts[i] = round(weights[i] * (data.n_cols - gaussians));
    // Ensure one point minimum in each.
    counts += 1;

    // Account for rounding errors (possibly necessary).
    counts[gaussians - 1] += (data.n_cols - arma::accu(counts));

    // Build each Gaussian individually.
    size_t point = 0;
    for (size_t i = 0; i < gaussians; ++i)
    {
      arma::mat gaussian;
      gaussian.randn(dims, counts[i]);

      // Randomly generate mean and covariance.
      means[i].randu(dims);
      means[i] -= 0.5;
      means[i] *= 50;

      // We need to make sure the covariance is positive definite.  We will take
      // a random matrix C and then set our covariance to 4 * C * C', which will
      // be positive semidefinite.
      covars[i].randu(dims, dims);
      covars[i] *= 4 * trans(covars[i]);

      data.cols(point, point + counts[i] - 1) = (covars[i] * gaussian + means[i]
          * arma::ones<arma::rowvec>(counts[i]));

      // Calculate the actual means and covariances because they will probably
      // be different (this is easier to do before we shuffle the points).
      means[i] = arma::mean(data.cols(point, point + counts[i] - 1), 1);
      covars[i] = ColumnCovariance(arma::mat(data.cols(point,
          point + counts[i] - 1)), 1 /* biased */);

      point += counts[i];
    }

    // Calculate actual weights.
    for (size_t i = 0; i < gaussians; ++i)
      weights[i] = (double) counts[i] / data.n_cols;

    // Now train the model.
    GMM gmm(gaussians, dims);
    gmm.Train(data, 10);

    arma::uvec sortRef = sort_index(weights);
    arma::uvec sortTry = sort_index(gmm.Weights());

    // If it's a bad match, try training again with a different seed.  We
    // probably just fell into some bad local minimum or had a bad starting
    // point.
    gmm = GMM(gaussians, dims);
    gmm.Train(data, 10);

    sortTry = sort_index(gmm.Weights());

    if (arma::norm(weights.elem(sortRef) - gmm.Weights().elem(sortTry)) > 0.1)
      continue;

    // Check the model to see that it is correct.
    for (size_t i = 0; i < gaussians; ++i)
    {
      // Check the mean.
      REQUIRE(
          arma::norm(gmm.Component(sortTry[i]).Mean() - means[sortRef[i]]) <
          0.05);
      // Check the covariance.
      REQUIRE(
          arma::norm(gmm.Component(sortTry[i]).Covariance() -
                                   covars[sortRef[i]]) < 0.2);
      // Check the weight.
      REQUIRE(gmm.Weights()[sortTry[i]] ==
          Approx(weights[sortRef[i]]).epsilon(0.00005));
    }

    success = true;
    break; // No need for multiple iterations.
  }

  REQUIRE(success == true);
}

/**
 * Train a single-gaussian mixture, but using the overload of Train() where
 * probabilities of the observation are given.
 */
TEST_CASE("GMMTrainEMSingleGaussianWithProbability", "[GMMTest]")
{
  // We run the test multiple times, since it sometimes fails, in order to get
  // the probability of failure down.
  bool success = false;
  const size_t trials = 3;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    // Generate observations from a Gaussian distribution.
    GaussianDistribution d("0.5 1.0", "1.0 0.3; 0.3 1.0");

    // 10000 observations, each with random probability.
    arma::mat observations(2, 20000);
    for (size_t i = 0; i < 20000; ++i)
      observations.col(i) = d.Random();
    arma::vec probabilities;
    probabilities.randu(20000); // Random probabilities.

    // Now train the model.
    GMM g(1, 2);
    g.Train(observations, probabilities, 10);

    // Check that it is trained correctly.  5% tolerance because of random error
    // present in observations.
    if (g.Component(0).Mean()[0] == Approx(0.5).epsilon(0.05) &&
        g.Component(0).Mean()[1] == Approx(1.0).epsilon(0.05) &&
        g.Component(0).Covariance()(0, 0) == Approx(1.0).epsilon(0.06) &&
        g.Component(0).Covariance()(0, 1) == Approx(0.3).epsilon(0.1) &&
        g.Component(0).Covariance()(1, 0) == Approx(0.3).epsilon(0.1) &&
        g.Component(0).Covariance()(1, 1) == Approx(1.0).epsilon(0.06) &&
        g.Weights()[0] == Approx(1.0).epsilon(1e-7))
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Train a multi-Gaussian mixture, using the overload of Train() where
 * probabilities of the observation are given.
 */
TEST_CASE("GMMTrainEMMultipleGaussiansWithProbability", "[GMMTest]")
{
  // We'll have three Gaussian distributions from this mixture, and one Gaussian
  // not from this mixture (but we'll put some observations from it in).
  GaussianDistribution d1("0.0 1.0 0.0", "1.0 0.0 0.5;"
                                         "0.0 0.8 0.1;"
                                         "0.5 0.1 1.0");
  GaussianDistribution d2("2.0 -1.0 5.0", "3.0 0.0 0.5;"
                                          "0.0 1.2 0.2;"
                                          "0.5 0.2 1.3");
  GaussianDistribution d3("0.0 5.0 -3.0", "2.0 0.0 0.0;"
                                          "0.0 0.3 0.0;"
                                          "0.0 0.0 1.0");
  GaussianDistribution d4("4.0 2.0 2.0", "1.5 0.6 0.5;"
                                         "0.6 1.1 0.1;"
                                         "0.5 0.1 1.0");

  // Now we'll generate points and probabilities.  2000 points.  Slower than I
  // would like...
  arma::mat points(3, 2000);
  arma::vec probabilities(2000);

  for (size_t i = 0; i < 2000; ++i)
  {
    double randValue = Random();

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
    double confidence = 0.998;
    double perturbation = Random(-0.002, 0.002);

    if (randValue <= 0.90)
      probabilities(i) = confidence + perturbation;
    else
      probabilities(i) = (1 - confidence) + perturbation;
  }

  // Now train the model.
  GMM g(3, 3); // 3 dimensions, 3 components (the fourth component is fake).

  EMFit<> fitter(100, 1e-5);
  g.Train(points, probabilities, 3, false, fitter);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(g.Weights());

  // The tolerances in our checks are quite large, but it is good to remember
  // that we introduced a fair amount of random noise into this whole process.
  // We don't need to look for the fourth Gaussian since that is not supposed to
  // be a part of this mixture.

  // First Gaussian (d1).
  REQUIRE(g.Weights()[sortedIndices[0]] - 0.2 == Approx(0.0).margin(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE((g.Component(sortedIndices[0]).Mean()[i]
        - d1.Mean()[i]) == Approx(0.0).margin(0.4));
  }

  for (size_t row = 0; row < 3; row++)
  {
    for (size_t col = 0; col < 3; col++)
    {
      // Big tolerance!  Lots of noise.
      REQUIRE((g.Component(sortedIndices[0]).Covariance()(row, col)
          - d1.Covariance()(row, col)) == Approx(0.0).margin(0.7));
    }
  }

  // Second Gaussian (d2).
  REQUIRE(g.Weights()[sortedIndices[1]] - 0.3 == Approx(0.0).margin(0.1));

  for (size_t i = 0; i < 3; ++i)
    REQUIRE((g.Component(sortedIndices[1]).Mean()[i]
        - d2.Mean()[i]) == Approx(0.0).margin(0.4));

  for (size_t row = 0; row < 3; row++)
  {
    for (size_t col = 0; col < 3; col++)
    {
      // Big tolerance!  Lots of noise.
      REQUIRE((g.Component(sortedIndices[1]).Covariance()(row, col)
          - d2.Covariance()(row, col)) == Approx(0.0).margin(0.7));
    }
  }

  // Third Gaussian (d3).
  REQUIRE(g.Weights()[sortedIndices[2]] - 0.4 == Approx(0.0).margin(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE((g.Component(sortedIndices[2]).Mean()[i]
        - d3.Mean()[i]) == Approx(0.0).margin(0.4));
  }

  for (size_t row = 0; row < 3; ++row)
  {
    for (size_t col = 0; col < 3; ++col)
    {
      REQUIRE((g.Component(sortedIndices[2]).Covariance()(row, col)
          - d3.Covariance()(row, col)) == Approx(0.0).margin(0.7));
    }
  }
}

/**
 * Make sure generating observations randomly works.  We'll do this by
 * generating a bunch of random observations and then re-training on them, and
 * hope that our model is the same.
 */
TEST_CASE("GMMRandomTest", "[GMMTest]")
{
  // Simple GMM distribution.
  GMM gmm(2, 2);
  gmm.Weights() = arma::vec("0.40 0.60");

  // N([2.25 3.10], [1.00 0.20; 0.20 0.89])
  gmm.Component(0) = GaussianDistribution("2.25 3.10", "1.00 0.60; 0.60 0.89");


  // N([4.10 1.01], [1.00 0.00; 0.00 1.01])
  gmm.Component(1) = GaussianDistribution("4.10 1.01", "1.00 0.70; 0.70 1.01");

  // Now generate a bunch of observations.
  arma::mat observations(2, 4000);
  for (size_t i = 0; i < 4000; ++i)
    observations.col(i) = gmm.Random();

  // A new one which we'll train.
  GMM gmm2(2, 2);
  gmm2.Train(observations, 10);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(gmm2.Weights());

  // Now check that the parameters are the same.  Tolerances are kind of big
  // because we only used 2000 observations.
  REQUIRE(gmm.Weights()[0] ==
      Approx(gmm2.Weights()[sortedIndices[0]]).epsilon(0.07));
  REQUIRE(gmm.Weights()[1] ==
      Approx(gmm2.Weights()[sortedIndices[1]]).epsilon(0.07));

  REQUIRE(gmm.Component(0).Mean()[0] ==
      Approx(gmm2.Component(sortedIndices[0]).Mean()[0]).epsilon(0.075));
  REQUIRE(gmm.Component(0).Mean()[1] ==
      Approx(gmm2.Component(sortedIndices[0]).Mean()[1]).epsilon(0.075));

  REQUIRE(gmm.Component(0).Covariance()(0, 0) ==
      Approx(gmm2.Component(sortedIndices[0]).Covariance()(0,
      0)).epsilon(0.13));
  REQUIRE(gmm.Component(0).Covariance()(0, 1) ==
      Approx(gmm2.Component(sortedIndices[0]).Covariance()(0,
      1)).epsilon(0.22));
  REQUIRE(gmm.Component(0).Covariance()(1, 0) ==
      Approx(gmm2.Component(sortedIndices[0]).Covariance()(1,
      0)).epsilon(0.22));
  REQUIRE(gmm.Component(0).Covariance()(1, 1) ==
      Approx(gmm2.Component(sortedIndices[0]).Covariance()(1,
      1)).epsilon(0.13));

  REQUIRE(gmm.Component(1).Mean()[0] ==
      Approx(gmm2.Component(sortedIndices[1]).Mean()[0]).epsilon(0.075));
  REQUIRE(gmm.Component(1).Mean()[1] ==
      Approx(gmm2.Component(sortedIndices[1]).Mean()[1]).epsilon(0.075));

  REQUIRE(gmm.Component(1).Covariance()(0, 0) ==
      Approx(gmm2.Component(sortedIndices[1]).Covariance()(0,
      0)).epsilon(0.13));
  REQUIRE(gmm.Component(1).Covariance()(0, 1) ==
      Approx(gmm2.Component(sortedIndices[1]).Covariance()(0,
      1)).epsilon(0.22));
  REQUIRE(gmm.Component(1).Covariance()(1, 0) ==
      Approx(gmm2.Component(sortedIndices[1]).Covariance()(1,
      0)).epsilon(0.22));
  REQUIRE(gmm.Component(1).Covariance()(1, 1) ==
      Approx(gmm2.Component(sortedIndices[1]).Covariance()(1,
      1)).epsilon(0.13));
}

/**
 * Test classification of observations by component.
 */
TEST_CASE("GMMClassifyTest", "[GMMTest]")
{
  // First create a Gaussian with a few components.
  GMM gmm(3, 2);
  gmm.Component(0) = GaussianDistribution("0 0", "1 0; 0 1");
  gmm.Component(1) = GaussianDistribution("1 3", "3 2; 2 3");
  gmm.Component(2) = GaussianDistribution("-2 -2", "2.2 1.4; 1.4 5.1");
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

  arma::Row<size_t> classes;

  gmm.Classify(observations, classes);

  // Test classification of points.  Classifications produced by hand.
  REQUIRE(classes[ 0] == 0);
  REQUIRE(classes[ 1] == 0);
  REQUIRE(classes[ 2] == 1);
  REQUIRE(classes[ 3] == 0);
  REQUIRE(classes[ 4] == 0);
  REQUIRE(classes[ 5] == 0);
  REQUIRE(classes[ 6] == 1);
  REQUIRE(classes[ 7] == 2);
  REQUIRE(classes[ 8] == 1);
  REQUIRE(classes[ 9] == 1);
  REQUIRE(classes[10] == 0);
  REQUIRE(classes[11] == 2);
  REQUIRE(classes[12] == 2);
}

TEST_CASE("GMMLoadSaveTest", "[GMMTest]")
{
  // Create a GMM, save it, and load it.
  GMM gmm(10, 4);
  gmm.Weights().randu();

  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    gmm.Component(i).Mean().randu();
    arma::mat covariance = arma::randu<arma::mat>(
        gmm.Component(i).Covariance().n_rows,
        gmm.Component(i).Covariance().n_cols);
    covariance *= covariance.t();
    covariance += arma::eye<arma::mat>(covariance.n_rows, covariance.n_cols);
    gmm.Component(i).Covariance(std::move(covariance));
  }

  // Save the GMM.
  {
    std::ofstream ofs("test-gmm-save.xml");
    cereal::XMLOutputArchive ar(ofs);
    ar(CEREAL_NVP(gmm));
  }

  // Load the GMM.
  GMM gmm2;
  {
    std::ifstream ifs("test-gmm-save.xml");
    cereal::XMLInputArchive ar(ifs);
    ar(cereal::make_nvp("gmm", gmm2));
  }

  // Remove clutter.
  // remove("test-gmm-save.xml");

  REQUIRE(gmm.Gaussians() == gmm2.Gaussians());
  REQUIRE(gmm.Dimensionality() == gmm2.Dimensionality());

  for (size_t i = 0; i < gmm.Dimensionality(); ++i)
    REQUIRE(gmm.Weights()[i] == Approx(gmm2.Weights()[i]).epsilon(1e-5));

  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      REQUIRE(gmm.Component(i).Mean()[j] ==
          Approx(gmm2.Component(i).Mean()[j]).epsilon(1e-5));
    }

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
      {
        REQUIRE(gmm.Component(i).Covariance()(j, k) ==
            Approx(gmm2.Component(i).Covariance()(j, k)).epsilon(1e-5));
      }
    }
  }
}

TEST_CASE("NoConstraintTest", "[GMMTest]")
{
  // Generate random matrices and make sure they end up the same.
  for (size_t i = 0; i < 30; ++i)
  {
    const size_t rows = 5 + RandInt(100);
    const size_t cols = 5 + RandInt(100);
    arma::mat cov(rows, cols);
    cov.randu();
    arma::mat newcov(cov);

    NoConstraint::ApplyConstraint(newcov);

    for (size_t j = 0; j < cov.n_elem; ++j)
      REQUIRE(newcov(j) == Approx(cov(j)).epsilon(1e-22));
  }
}

TEST_CASE("PositiveDefiniteConstraintTest", "[GMMTest]")
{
  // Make sure matrices are made to be positive definite, or more specifically,
  // that they can be Cholesky decomposed.
  for (size_t i = 0; i < 30; ++i)
  {
    const size_t elem = 5 + RandInt(50);
    arma::mat cov(elem, elem);
    cov.randu();

    PositiveDefiniteConstraint::ApplyConstraint(cov);

    arma::mat c;
    REQUIRE(arma::chol(c, cov, "lower"));
  }
}

TEST_CASE("DiagonalConstraintTest", "[GMMTest]")
{
  // Make sure matrices are made to be positive definite.
  for (size_t i = 0; i < 30; ++i)
  {
    const size_t elem = 5 + RandInt(50);
    arma::mat cov(elem, elem);
    cov.randu();

    DiagonalConstraint::ApplyConstraint(cov);

    for (size_t j = 0; j < elem; ++j)
      for (size_t k = 0; k < elem; ++k)
        if (j != k)
          REQUIRE(cov(j, k) == Approx(0.0).margin(1e-50));
  }
}

TEST_CASE("EigenvalueRatioConstraintTest", "[GMMTest]")
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
      REQUIRE(eigenvalues[i] / eigenvalues[0] ==
          Approx(ratios[i]).epsilon(1e-7));
  }
}

TEST_CASE("UseExistingModelTest", "[GMMTest]")
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

  for (size_t i = 0; i < gaussians; ++i)
    counts[i] = round(weights[i] * (data.n_cols - gaussians));
  // Ensure one point minimum in each.
  counts += 1;

  // Account for rounding errors (possibly necessary).
  counts[gaussians - 1] += (data.n_cols - arma::accu(counts));

  // Build each Gaussian individually.
  size_t point = 0;
  for (size_t i = 0; i < gaussians; ++i)
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
    covars[i] = ColumnCovariance(arma::mat(data.cols(point,
        point + counts[i] - 1)), 1 /* biased */);

    point += counts[i];
  }

  // Calculate actual weights.
  for (size_t i = 0; i < gaussians; ++i)
    weights[i] = (double) counts[i] / data.n_cols;

  // Now train the model.
  GMM gmm(gaussians, dims);
  gmm.Train(data, 10);

  GMM oldgmm(gmm);

  // Retrain the model with the existing model as the starting point.
  gmm.Train(data, 1, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    REQUIRE(gmm.Weights()[i] == Approx(oldgmm.Weights()[i]).epsilon(1e-4));

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      REQUIRE(gmm.Component(i).Mean()[j] ==
          Approx(oldgmm.Component(i).Mean()[j]).epsilon(1e-4));

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
      {
        REQUIRE(gmm.Component(i).Covariance()(j, k) ==
            Approx(oldgmm.Component(i).Covariance()(j, k)).epsilon(1e-4));
      }
    }
  }

  // Do it again, with a larger number of trials.
  gmm = oldgmm;

  // Retrain the model with the existing model as the starting point.
  gmm.Train(data, 10, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    REQUIRE(gmm.Weights()[i] == Approx(oldgmm.Weights()[i]).epsilon(1e-7));

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      REQUIRE(gmm.Component(i).Mean()[j] ==
            Approx(oldgmm.Component(i).Mean()[j]).epsilon(1e-5));

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
      {
        REQUIRE(gmm.Component(i).Covariance()(j, k) ==
            Approx(oldgmm.Component(i).Covariance()(j, k)).epsilon(1e-5));
      }
    }
  }

  // Do it again, but using the overload of Train() that takes probabilities
  // into account.
  arma::vec probabilities(data.n_cols);
  probabilities.ones(); // Fill with ones.

  gmm = oldgmm;
  gmm.Train(data, probabilities, 1, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    REQUIRE(gmm.Weights()[i] == Approx(oldgmm.Weights()[i]).epsilon(1e-7));

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      REQUIRE(gmm.Component(i).Mean()[j] ==
          Approx(oldgmm.Component(i).Mean()[j]).epsilon(1e-5));

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
      {
        REQUIRE(gmm.Component(i).Covariance()(j, k) ==
            Approx(oldgmm.Component(i).Covariance()(j, k)).epsilon(1e-5));
      }
    }
  }

  // One more time, with multiple trials.
  gmm = oldgmm;
  gmm.Train(data, probabilities, 10, true);

  // Check for similarity.
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    REQUIRE(gmm.Weights()[i] == Approx(oldgmm.Weights()[i]).epsilon(1e-7));

    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      REQUIRE(gmm.Component(i).Mean()[j] ==
          Approx(oldgmm.Component(i).Mean()[j]).epsilon(1e-5));

      for (size_t k = 0; k < gmm.Dimensionality(); ++k)
      {
        REQUIRE(gmm.Component(i).Covariance()(j, k) ==
            Approx(oldgmm.Component(i).Covariance()(j, k)).epsilon(1e-5));
      }
    }
  }
}

/********************************************************/
/** Diagonal Gaussian Mixture Model(DiagonalGMM) Tests **/
/********************************************************/

/**
 * Make sure Diagonal::Probability() of a specific Gaussian component works
 * correctly in single observation.
 */
TEST_CASE("DiagonalGMMProbabilityComponentTest", "[GMMTest]")
{
  // Create DiagonalGMM.
  DiagonalGMM gmm(2, 2);
  gmm.Component(0) = DiagonalGaussianDistribution("0 0", "1 1");
  gmm.Component(1) = DiagonalGaussianDistribution("2 3", "3 2");
  gmm.Weights() = "0.2 0.8";

  // The values are calculated using mlpack's GMM class.
  REQUIRE(gmm.Probability("0 0", 0) == Approx(0.0318309886184).epsilon(1e-7));
  REQUIRE(gmm.Probability("0 0", 1) == Approx(0.00281282202844).epsilon(1e-7));

  REQUIRE(gmm.Probability("1 1", 0) == Approx(0.0117099663049).epsilon(1e-7));
  REQUIRE(gmm.Probability("1 1", 1) == Approx(0.016186673172).epsilon(1e-7));

  REQUIRE(gmm.Probability("3 3", 0) ==
      Approx(3.92825606928e-06).epsilon(1e-7));
  REQUIRE(gmm.Probability("3 3", 1) == Approx(0.0439999395467).epsilon(1e-7));

  REQUIRE(gmm.Probability("2.6 3.2", 0) ==
      Approx(6.47659933818e-06).epsilon(1e-7));
  REQUIRE(gmm.Probability("2.6 3.2", 1) ==
      Approx(0.0484656319247).epsilon(1e-7));

  REQUIRE(gmm.Probability("-4.1 2.1", 0) ==
      Approx(7.85209733164e-07).epsilon(1e-7));
  REQUIRE(gmm.Probability("-4.1 2.1", 1) ==
      Approx(8.60082772711e-05).epsilon(1e-7));
}

/**
 * Make sure we can train a model on only one Gaussian (randomly generated)
 * in two dimensions.  We will vary the dataset size from small to large.
 * The EM algorithm is used for training the DiagonalGMM.
 */
TEST_CASE("DiagonalGMMTrainEMOneGaussian", "[GMMTest]")
{
  for (size_t iterations = 0; iterations < 4; iterations++)
  {
    // Determine random mean, covariance, and observations.
    arma::vec mean(2, arma::fill::randu);
    arma::vec covar(2, arma::fill::randu);
    arma::mat data(2, 150 * pow(10, (iterations / 3.0)), arma::fill::randn);

    // Now apply mean and covariance.
    data.row(0) *= covar(0);
    data.row(1) *= covar(1);

    data.row(0) += mean(0);
    data.row(1) += mean(1);

    // Now, train the model.
    DiagonalGMM gmm(1, 2);
    gmm.Train(data, 10);

    arma::vec actualMean = arma::mean(data, 1);
    arma::vec actualCovar = arma::diagvec(
        ColumnCovariance(data,
        1 /* biased estimator */));

    // Check the model to see that it is correct.
    CheckMatrices(gmm.Component(0).Mean(), actualMean);
    CheckMatrices(gmm.Component(0).Covariance(), actualCovar);

    REQUIRE(gmm.Weights()[0] == Approx(1.0).epsilon(1e-7));
  }
}

/**
 * Make sure we can train a single Gaussian Mixture Model with diagonal
 * covariance reasonably using Train() where probabilities of the observation
 * are given.  The EM algorithm is used for training the DiagonalGMM.
 */
TEST_CASE("DiagonalGMMTrainEMOneGaussianWithProbability", "[GMMTest]")
{
  // Generate a diagonal covariance gaussian distribution.
  DiagonalGaussianDistribution d("1.0 0.8", "1.0 2.0");

  // Generate 20000 observations, each with random probabilities.
  arma::mat observations(2, 20000);
  for (size_t i = 0; i < 20000; ++i)
    observations.col(i) = d.Random();

  // Random probabilities.
  arma::vec probabilities = arma::randu<arma::vec>(20000);

  // Create DiagonalGMM.
  DiagonalGMM gmm(1, 2);
  size_t trials = 10;

  // Train this model.
  gmm.Train(observations, probabilities, trials);

  // Check the model is trained correctly.
  // 10% tolerance, because of possible noise.
  REQUIRE(gmm.Component(0).Mean()[0] == Approx(1.0).epsilon(0.08));
  REQUIRE(gmm.Component(0).Mean()[1] == Approx(0.8).epsilon(0.08));

  // 6% tolerance, because of possible noise.
  REQUIRE(gmm.Component(0).Covariance()[0] == Approx(1.0).epsilon(0.06));
  REQUIRE(gmm.Component(0).Covariance()[1] == Approx(2.0).epsilon(0.06));

  REQUIRE(gmm.Weights()[0] == Approx(1.0).epsilon(1e-7));
}

/**
 * Make sure we can train multiple Gaussian Mixture Models with diagonal
 * covariance reasonably.
 * The EM algorithm is used for training the DiagonalGMM.
 */
TEST_CASE("DiagonalGMMTrainEMMultipleGaussians", "[GMMTest]")
{
  // We'll have three diagonal covariance Gaussian distributions from this
  // mixture.
  DiagonalGaussianDistribution d1("0.0 1.0 0.0", "1.0 0.8 1.0;");
  DiagonalGaussianDistribution d2("2.0 -1.0 5.0", "3.0 1.2 1.3;");
  DiagonalGaussianDistribution d3("0.0 5.0 -3.0", "2.0 0.3 1.0;");

  // Now we'll generate points and probabilities.
  arma::mat observations(3, 5000);

  for (size_t i = 0; i < 5000; ++i)
  {
    double randValue = Random();

    if (randValue <= 0.20) // p(d1) = 0.20
      observations.col(i) = d1.Random();
    else if (randValue <= 0.50) // p(d2) = 0.30
      observations.col(i) = d2.Random();
    else // p(d3) = 0.50
      observations.col(i) = d3.Random();
  }

  // Now train the model.  3 dimensions, 3 components.
  DiagonalGMM g(3, 3);
  size_t trials = 5;
  g.Train(observations, trials);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(g.Weights());

  // First Gaussian (d1).
  REQUIRE(g.Weights()[sortedIndices[0]] - 0.2 == Approx(0.0).margin(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE((g.Component(sortedIndices[0]).Mean()[i]
        - d1.Mean()[i]) == Approx(0.0).margin(0.4));
  }

  for (size_t i = 0; i < 3; ++i)
  {
    const double v = g.Component(sortedIndices[0]).Covariance()(i);
    REQUIRE(v - d1.Covariance()(i) == Approx(0.0).margin(0.5));
  }

  // Second Gaussian (d2).
  REQUIRE(g.Weights()[sortedIndices[1]] - 0.3 == Approx(0.0).margin(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE((g.Component(sortedIndices[1]).Mean()[i]
        - d2.Mean()[i]) == Approx(0.0).margin(0.4));
  }

  for (size_t i = 0; i < 3; ++i)
  {
    const double v = g.Component(sortedIndices[1]).Covariance()(i);
    REQUIRE(v - d2.Covariance()(i) == Approx(0.0).margin(0.5));
  }

  // Third Gaussian (d3).
  REQUIRE(g.Weights()[sortedIndices[2]] - 0.5 == Approx(0.0).margin(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE((g.Component(sortedIndices[2]).Mean()[i]
        - d3.Mean()[i]) == Approx(0.0).margin(0.4));
  }

  for (size_t i = 0; i < 3; ++i)
  {
    const double v = g.Component(sortedIndices[2]).Covariance()(i);
    REQUIRE(v - d3.Covariance()(i) == Approx(0.0).margin(0.5));
  }
}

/**
 * Make sure we can train multiple Gaussian Mixture Models with diagonal
 * covariance reasonably using Train() where probabilities of the observation
 * are given.  The EM algorithm is used for training the DiagonalGMM.
 */
TEST_CASE("DiagonalGMMTrainEMMultipleGaussiansWithProbability", "[GMMTest]")
{
  // We'll have three diagonal covariance Gaussian distributions from this
  // mixture.
  DiagonalGaussianDistribution d1("1.5 0.8 1.0", "1.0 0.8 1.0;");
  DiagonalGaussianDistribution d2("8.2 6.3 7.4", "1.0 1.2 1.3;");
  DiagonalGaussianDistribution d3("-4.5 -5.0 -3.0", "2.0 2.3 1.0;");

  // Now we'll generate observations and probabilities.
  arma::mat observations(3, 10000);

  for (size_t i = 0; i < 10000; ++i)
  {
    double randValue = Random();

    if (randValue <= 0.20) // p(d1) = 0.20
      observations.col(i) = d1.Random();
    else if (randValue <= 0.50) // p(d2) = 0.30
      observations.col(i) = d2.Random();
    else // p(d3) = 0.50
      observations.col(i) = d3.Random();
  }

  // Random probabilities.
  arma::vec probabilities = arma::randu<arma::vec>(10000);

  // Now train the model.  3 gaussians, 3 dimensions.
  DiagonalGMM g(3, 3);
  size_t trials = 5;
  g.Train(observations, probabilities, trials);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(g.Weights());

  // First Gaussian (d1).
  REQUIRE(g.Weights()[sortedIndices[0]] == Approx(0.2).epsilon(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE(g.Component(sortedIndices[0]).Mean()[i] ==
        Approx(d1.Mean()[i]).epsilon(0.13));
  }

  for (size_t i = 0; i < 3; ++i)
  {
    const double v = g.Component(sortedIndices[0]).Covariance()(i);
    REQUIRE(v == Approx(d1.Covariance()(i)).epsilon(0.17));
  }

  // Second Gaussian (d2).
  REQUIRE(g.Weights()[sortedIndices[1]] == Approx(0.3).epsilon(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE(g.Component(sortedIndices[1]).Mean()[i] ==
        Approx(d2.Mean()[i]).epsilon(0.13));
  }

  for (size_t i = 0; i < 3; ++i)
  {
    const double v = g.Component(sortedIndices[1]).Covariance()(i);
    REQUIRE(v == Approx(d2.Covariance()(i)).epsilon(0.17));
  }

  // Third Gaussian (d3).
  REQUIRE(g.Weights()[sortedIndices[2]] == Approx(0.5).epsilon(0.1));

  for (size_t i = 0; i < 3; ++i)
  {
    REQUIRE(g.Component(sortedIndices[2]).Mean()[i] ==
        Approx(d3.Mean()[i]).epsilon(0.13));
  }

  for (size_t i = 0; i < 3; ++i)
  {
    const double v = g.Component(sortedIndices[2]).Covariance()(i);
    REQUIRE(v == Approx(d3.Covariance()(i)).epsilon(0.17));
  }
}

/**
 * Make sure generating observations randomly works.  We'll do this by
 * generating a bunch of random observations and then re-training on them, and
 * hope that our model is the same.
 */
TEST_CASE("DiagonalGMMRandomTest", "[GMMTest]")
{
  // Simple GMM distribution.
  DiagonalGMM gmm(2, 2);
  gmm.Weights() = arma::vec("0.40 0.60");

  gmm.Component(0) = DiagonalGaussianDistribution("1.05 2.60", "0.95 1.01");
  gmm.Component(1) = DiagonalGaussianDistribution("4.30 1.00", "1.05 0.97");

  // Now generate a bunch of observations.
  arma::mat observations(2, 4000);
  for (size_t i = 0; i < 4000; ++i)
    observations.col(i) = gmm.Random();

  // A new one which we'll train.
  DiagonalGMM gmm2(2, 2);
  gmm2.Train(observations, 10);

  // Now check the results.  We need to order by weights so that when we do the
  // checking, things will be correct.
  arma::uvec sortedIndices = sort_index(gmm2.Weights());

  // Check that the parameters are the same. Tolerances vary,
  // because of possible noise.
  REQUIRE(gmm.Weights()[0] ==
      Approx(gmm2.Weights()[sortedIndices[0]]).epsilon(0.09));
  REQUIRE(gmm.Weights()[1] ==
      Approx(gmm2.Weights()[sortedIndices[1]]).epsilon(0.09));

  // Check the means are the same.
  REQUIRE(gmm.Component(0).Mean()[0] ==
      Approx(gmm2.Component(sortedIndices[0]).Mean()[0]).epsilon(0.13));
  REQUIRE(gmm.Component(0).Mean()[1] ==
      Approx(gmm2.Component(sortedIndices[0]).Mean()[1]).epsilon(0.13));

  REQUIRE(gmm.Component(1).Mean()[0] ==
      Approx(gmm2.Component(sortedIndices[1]).Mean()[0]).epsilon(0.13));
  REQUIRE(gmm.Component(1).Mean()[1] ==
      Approx(gmm2.Component(sortedIndices[1]).Mean()[1]).epsilon(0.13));


  // Check the covariances are the same.
  REQUIRE(gmm.Component(0).Covariance()(0) ==
      Approx(gmm2.Component(sortedIndices[0]).Covariance()(0)).epsilon(0.22));
  REQUIRE(gmm.Component(0).Covariance()(1) ==
      Approx(gmm2.Component(sortedIndices[0]).Covariance()(1)).epsilon(0.22));

  REQUIRE(gmm.Component(1).Covariance()(0) ==
      Approx(gmm2.Component(sortedIndices[1]).Covariance()(0)).epsilon(0.22));
  REQUIRE(gmm.Component(1).Covariance()(1) ==
      Approx(gmm2.Component(sortedIndices[1]).Covariance()(1)).epsilon(0.22));
}

//! Make sure load and save DiagonalGMM correctly.
TEST_CASE("DiagonalGMMLoadSaveTest", "[GMMTest]")
{
  // Create a DiagonalGMM, save and load it.
  DiagonalGMM gmm(10, 4);
  gmm.Weights().randu();

  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    gmm.Component(i).Mean().randu();
    arma::vec covariance = arma::randu<arma::vec>(
        gmm.Component(i).Covariance().n_elem);

    gmm.Component(i).Covariance(std::move(covariance));
  }

  // Save the gmm.
  {
    std::ofstream ofs("test-diagonal-gmm-save.xml");
    cereal::XMLOutputArchive ar(ofs);
    ar(cereal::make_nvp("gmm", gmm));
  }

  // Load the gmm into gmm2.
  DiagonalGMM gmm2;
  {
    std::ifstream ifs("test-diagonal-gmm-save.xml");
    cereal::XMLInputArchive ar(ifs);
    ar(cereal::make_nvp("gmm", gmm2));
  }

  // Remove clutter.
  remove("test-diagonal-gmm-save.xml");

  // Check the parameters are the same.
  REQUIRE(gmm.Gaussians() == gmm2.Gaussians());
  REQUIRE(gmm.Dimensionality() == gmm2.Dimensionality());

  for (size_t i = 0; i < gmm.Dimensionality(); ++i)
    REQUIRE(gmm.Weights()[i] == Approx(gmm2.Weights()[i]).epsilon(1e-7));

  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    for (size_t j = 0; j < gmm.Dimensionality(); ++j)
    {
      REQUIRE(gmm.Component(i).Mean()[j] ==
          Approx(gmm2.Component(i).Mean()[j]).epsilon(1e-5));
      REQUIRE(gmm.Component(i).Covariance()(j) ==
          Approx(gmm2.Component(i).Covariance()(j)).epsilon(1e-5));
    }
  }
}
