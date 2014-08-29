/**
 * @file pca_test.cpp
 * @author Ajinkya Kale
 *
 * Test file for PCA class.
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
#include <mlpack/methods/pca/pca.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(PCATest);

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::pca;
using namespace mlpack::distribution;

/**
 * Compare the output of our PCA implementation with Armadillo's.
 */
BOOST_AUTO_TEST_CASE(ArmaComparisonPCATest)
{
  mat coeff, coeff1;
  vec eigVal, eigVal1;
  mat score, score1;

  mat data = randu<mat>(100, 100);

  PCA p;

  p.Apply(data, score1, eigVal1, coeff1);
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

/**
 * Test that dimensionality reduction with PCA works the same way MATLAB does
 * (which should be correct!).
 */
BOOST_AUTO_TEST_CASE(PCADimensionalityReductionTest)
{
  // Fake, simple dataset.  The results we will compare against are from MATLAB.
  mat data("1 0 2 3 9;"
           "5 2 8 4 8;"
           "6 7 3 1 8");

  // Now run PCA to reduce the dimensionality.
  PCA p;
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
 * reduction works.
 */
BOOST_AUTO_TEST_CASE(PCAVarianceRetainedTest)
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
  PCA p;
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
