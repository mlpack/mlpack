/**
 * @file pca_test.cpp
 * @author Ajinkya Kale
 * @author Marcus Edel
 *
 * Test file for PCA class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

BOOST_AUTO_TEST_SUITE(PCATest);

using namespace arma;
using namespace mlpack;
using namespace mlpack::pca;
using namespace mlpack::distribution;

/*
 * Compare the output of the our PCA implementation with Armadillo's using the
 * specified decomposition policy.
 */
template<typename DecompositionPolicy>
void ArmaComparisonPCA()
{
  arma::mat coeff, coeff1, score, score1;
  arma::vec eigVal, eigVal1;

  arma::mat data = arma::randu<arma::mat>(3, 1000);

  PCAType<DecompositionPolicy> exactPCA;
  exactPCA.Apply(data, score1, eigVal1, coeff1);

  princomp(coeff, score, eigVal, trans(data));

  // Verify the PCA results based on the eigenvalues.
  for (size_t i = 0; i < eigVal.n_elem; i++)
  {
    if (eigVal[i] == 0.0)
      BOOST_REQUIRE_SMALL(eigVal1[i], 1e-15);
    else
      BOOST_REQUIRE_CLOSE(eigVal[i], eigVal1[i], 0.0001);
  }
}

/*
 * Test that dimensionality reduction with PCA works the same way MATLAB does
 * (which should be correct!) using the specified decomposition policy.
 */
template<typename DecompositionPolicy>
void PCADimensionalityReduction()
{
  // Fake, simple dataset.  The results we will compare against are from MATLAB.
  mat data("1 0 2 3 9;"
           "5 2 8 4 8;"
           "6 7 3 1 8");

  // Now run PCA to reduce the dimensionality.
  PCAType<DecompositionPolicy> p;
  const double varRetained = p.Apply(data, 2); // Reduce to 2 dimensions.

  // Compare with correct results.
  mat correct("-1.53781086 -3.51358020 -0.16139887 -1.87706634  7.08985628;"
              " 1.29937798  3.45762685 -2.69910005 -3.15620704  1.09830225");

  BOOST_REQUIRE_EQUAL(data.n_rows, correct.n_rows);
  BOOST_REQUIRE_EQUAL(data.n_cols, correct.n_cols);

  // If the eigenvectors are pointed opposite directions, they will cancel
  // each other out in this summation.
  for (size_t i = 0; i < data.n_rows; i++)
  {
    if (accu(abs(correct.row(i) + data.row(i))) < 0.001 /* arbitrary */)
    {
      // Flip Armadillo coefficients for this column.
      data.row(i) *= -1;
    }
  }

  for (size_t row = 0; row < 2; row++)
    for (size_t col = 0; col < 5; col++)
      BOOST_REQUIRE_CLOSE(data(row, col), correct(row, col), 1e-3);

  // Check that the amount of variance retained is right.
  BOOST_REQUIRE_CLOSE(varRetained, 0.904876047045906, 1e-5);
}

/**
 * Test that setting the variance retained parameter to perform dimensionality
 * reduction works using the specified decomposition policy.
 */
template<typename DecompositionPolicy>
void PCAVarianceRetained()
{
    // Fake, simple dataset.
  mat data("1 0 2 3 9;"
           "5 2 8 4 8;"
           "6 7 3 1 8");

  // The normalized eigenvalues:
  //   0.616237391936100
  //   0.288638655109805
  //   0.095123952954094
  // So if we keep one dimension, the actual variance retained is
  //   0.616237391936100
  // and if we keep two, the actual variance retained is
  //   0.904876047045906
  // and if we keep three, the actual variance retained is 1.
  PCAType<DecompositionPolicy> p;
  arma::mat origData = data;
  double varRetained = p.Apply(data, 0.1);

  BOOST_REQUIRE_EQUAL(data.n_rows, 1);
  BOOST_REQUIRE_EQUAL(data.n_cols, 5);
  BOOST_REQUIRE_CLOSE(varRetained, 0.616237391936100, 1e-5);

  data = origData;
  varRetained = p.Apply(data, 0.5);

  BOOST_REQUIRE_EQUAL(data.n_rows, 1);
  BOOST_REQUIRE_EQUAL(data.n_cols, 5);
  BOOST_REQUIRE_CLOSE(varRetained, 0.616237391936100, 1e-5);

  data = origData;
  varRetained = p.Apply(data, 0.7);

  BOOST_REQUIRE_EQUAL(data.n_rows, 2);
  BOOST_REQUIRE_EQUAL(data.n_cols, 5);
  BOOST_REQUIRE_CLOSE(varRetained, 0.904876047045906, 1e-5);

  data = origData;
  varRetained = p.Apply(data, 0.904);

  BOOST_REQUIRE_EQUAL(data.n_rows, 2);
  BOOST_REQUIRE_EQUAL(data.n_cols, 5);
  BOOST_REQUIRE_CLOSE(varRetained, 0.904876047045906, 1e-5);

  data = origData;
  varRetained = p.Apply(data, 0.905);

  BOOST_REQUIRE_EQUAL(data.n_rows, 3);
  BOOST_REQUIRE_EQUAL(data.n_cols, 5);
  BOOST_REQUIRE_CLOSE(varRetained, 1.0, 1e-5);

  data = origData;
  varRetained = p.Apply(data, 1.0);

  BOOST_REQUIRE_EQUAL(data.n_rows, 3);
  BOOST_REQUIRE_EQUAL(data.n_cols, 5);
  BOOST_REQUIRE_CLOSE(varRetained, 1.0, 1e-5);
}

/**
 * Compare the output of our exact PCA implementation with Armadillo's.
 */
BOOST_AUTO_TEST_CASE(ArmaComparisonExactPCATest)
{
  ArmaComparisonPCA<ExactSVDPolicy>();
}

/**
 * Compare the output of our randomized-SVD PCA implementation with Armadillo's.
 */
BOOST_AUTO_TEST_CASE(ArmaComparisonRandomizedPCATest)
{
  ArmaComparisonPCA<RandomizedSVDPolicy>();
}

/**
 * Test that dimensionality reduction with exact-svd PCA works the same way
 * MATLAB does (which should be correct!).
 */
BOOST_AUTO_TEST_CASE(ExactPCADimensionalityReductionTest)
{
  PCADimensionalityReduction<ExactSVDPolicy>();
}

/**
 * Test that dimensionality reduction with randomized-svd PCA works the same way
 * MATLAB does (which should be correct!).
 */
BOOST_AUTO_TEST_CASE(RandomizedPCADimensionalityReductionTest)
{
  PCADimensionalityReduction<RandomizedSVDPolicy>();
}

/**
 * Test that dimensionality reduction with QUIC-SVD PCA works the same way
 * as the Exact-SVD PCA method.
 */
BOOST_AUTO_TEST_CASE(QUICPCADimensionalityReductionTest)
{
  arma::mat data, data1;
  data::Load("test_data_3_1000.csv", data);
  data1 = data;

  // It isn't guaranteed that the QUIC-SVD will match with the exact SVD method,
  // starting with random samples. If this works 1 of 5 times, I'm fine with
  // that. All I want to know is that the QUIC-SVD method is  able to solve the
  // task and is at least as good as the exact method (plus a little bit for
  // noise).
  size_t successes = 0;
  for (size_t trial = 0; trial < 5; ++trial)
  {

    PCAType<ExactSVDPolicy> exactPCA;
    const double varRetainedExact = exactPCA.Apply(data, 1);

    PCAType<QUICSVDPolicy> quicPCA;
    const double varRetainedQUIC = quicPCA.Apply(data1, 1);


    if (std::abs(varRetainedExact - varRetainedQUIC) < 0.2)
    {
      ++successes;
      break;
    }
  }

  BOOST_REQUIRE_GE(successes, 1);
  BOOST_REQUIRE_EQUAL(data.n_rows, data1.n_rows);
  BOOST_REQUIRE_EQUAL(data.n_cols, data1.n_cols);
}

/**
 * Test that setting the variance retained parameter to perform dimensionality
 * reduction works using the exact svd PCA method.
 */
BOOST_AUTO_TEST_CASE(ExactPCAVarianceRetainedTest)
{
  PCAVarianceRetained<ExactSVDPolicy>();
}

/**
 * Test that scaling PCA works.
 */
BOOST_AUTO_TEST_CASE(PCAScalingTest)
{
  // Generate an artificial dataset in 3 dimensions.
  arma::mat data(3, 5000);

  arma::vec mean("1.0 3.0 -12.0");
  arma::mat cov("1.0 0.9 0.0;"
                "0.9 1.0 0.0;"
                "0.0 0.0 12.0");
  GaussianDistribution g(mean, cov);

  for (size_t i = 0; i < 5000; ++i)
    data.col(i) = g.Random();

  // Now get the principal components when we are scaling.
  PCA p(true);
  arma::mat transData;
  arma::vec eigval;
  arma::mat eigvec;

  p.Apply(data, transData, eigval, eigvec);

  // The first two components of the eigenvector with largest eigenvalue should
  // be somewhere near sqrt(2) / 2.  The third component should be close to
  // zero.  There is noise, of course...
  BOOST_REQUIRE_CLOSE(std::abs(eigvec(0, 0)), sqrt(2) / 2, 0.2);
  BOOST_REQUIRE_CLOSE(std::abs(eigvec(1, 0)), sqrt(2) / 2, 0.2);
  BOOST_REQUIRE_SMALL(eigvec(2, 0), 0.08); // Large tolerance for noise.

  // The second component should be focused almost entirely in the third
  // dimension.
  BOOST_REQUIRE_SMALL(eigvec(0, 1), 0.08);
  BOOST_REQUIRE_SMALL(eigvec(1, 1), 0.08);
  BOOST_REQUIRE_CLOSE(std::abs(eigvec(2, 1)), 1.0, 0.2);

  // The third component should have the same absolute value characteristics as
  // the first.
  BOOST_REQUIRE_CLOSE(std::abs(eigvec(0, 0)), sqrt(2) / 2, 0.2); // 20% tolerance.
  BOOST_REQUIRE_CLOSE(std::abs(eigvec(1, 0)), sqrt(2) / 2, 0.2);
  BOOST_REQUIRE_SMALL(eigvec(2, 0), 0.08); // Large tolerance for noise.

  // The eigenvalues should sum to three.
  BOOST_REQUIRE_CLOSE(accu(eigval), 3.0, 0.1); // 10% tolerance.
}


BOOST_AUTO_TEST_SUITE_END();
