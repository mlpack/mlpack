/**
 * @file tests/pca_test.cpp
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
#include <mlpack/methods/pca.hpp>

#include "catch.hpp"

using namespace arma;
using namespace mlpack;

/*
 * Compare the output of the our PCA implementation with Armadillo's using the
 * specified decomposition policy.
 */
template<typename DecompositionPolicy>
void ArmaComparisonPCA(
    const bool scaleData = false,
    const DecompositionPolicy& decomposition = DecompositionPolicy())
{
  arma::mat coeff, coeff1, score, score1;
  arma::vec eigVal, eigVal1;

  arma::mat data = arma::randu<arma::mat>(3, 1000);

  PCA<DecompositionPolicy> pcaType(scaleData, decomposition);
  pcaType.Apply(data, score1, eigVal1, coeff1);

  princomp(coeff, score, eigVal, trans(data));

  // Verify the PCA results based on the eigenvalues.
  for (size_t i = 0; i < eigVal.n_elem; ++i)
  {
    if (eigVal[i] == 0.0)
      REQUIRE(eigVal1[i] == Approx(0.0).margin(1e-15));
    else
      REQUIRE(eigVal[i] == Approx(eigVal1[i]).epsilon(1e-6));
  }
}

/*
 * Test that dimensionality reduction with PCA works the same way MATLAB does
 * (which should be correct!) using the specified decomposition policy.
 */
template<typename DecompositionPolicy>
void PCADimensionalityReduction(
    const bool scaleData = false,
    const DecompositionPolicy& decomposition = DecompositionPolicy())
{
  // Fake, simple dataset.  The results we will compare against are from MATLAB.
  mat data("1 0 2 3 9;"
           "5 2 8 4 8;"
           "6 7 3 1 8");

  // Now run PCA to reduce the dimensionality.
  size_t trial = 0;
  bool success = false;
  double varRetained = 0.0;
  while (trial < 3 && !success)
  {
    // In some cases the LU decomposition may fail.
    try
    {
      PCA<DecompositionPolicy> p(scaleData, decomposition);
      varRetained = p.Apply(data, 2); // Reduce to 2 dimensions.
      success = true;
    }
    catch (std::logic_error&) { }

    ++trial;
  }

  REQUIRE(success == true);

  // Compare with correct results.
  mat correct("-1.53781086 -3.51358020 -0.16139887 -1.87706634  7.08985628;"
              " 1.29937798  3.45762685 -2.69910005 -3.15620704  1.09830225");

  REQUIRE(data.n_rows == correct.n_rows);
  REQUIRE(data.n_cols == correct.n_cols);

  // If the eigenvectors are pointed opposite directions, they will cancel
  // each other out in this summation.
  for (size_t i = 0; i < data.n_rows; ++i)
  {
    if (accu(abs(correct.row(i) + data.row(i))) < 0.001 /* arbitrary */)
    {
      // Flip Armadillo coefficients for this column.
      data.row(i) *= -1;
    }
  }

  for (size_t row = 0; row < 2; row++)
    for (size_t col = 0; col < 5; col++)
      REQUIRE(data(row, col) == Approx(correct(row, col)).epsilon(1e-5));

  // Check that the amount of variance retained is right.
  REQUIRE(varRetained == Approx(0.904876047045906).epsilon(1e-7));
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
  PCA<DecompositionPolicy> p;
  arma::mat origData = data;
  double varRetained = p.Apply(data, 0.1);

  REQUIRE(data.n_rows == 1);
  REQUIRE(data.n_cols == 5);
  REQUIRE(varRetained == Approx(0.616237391936100).epsilon(1e-7));

  data = origData;
  varRetained = p.Apply(data, 0.5);

  REQUIRE(data.n_rows == 1);
  REQUIRE(data.n_cols == 5);
  REQUIRE(varRetained == Approx(0.616237391936100).epsilon(1e-7));

  data = origData;
  varRetained = p.Apply(data, 0.7);

  REQUIRE(data.n_rows == 2);
  REQUIRE(data.n_cols == 5);
  REQUIRE(varRetained == Approx(0.904876047045906).epsilon(1e-7));

  data = origData;
  varRetained = p.Apply(data, 0.904);

  REQUIRE(data.n_rows == 2);
  REQUIRE(data.n_cols == 5);
  REQUIRE(varRetained == Approx(0.904876047045906).epsilon(1e-7));

  data = origData;
  varRetained = p.Apply(data, 0.905);

  REQUIRE(data.n_rows == 3);
  REQUIRE(data.n_cols == 5);
  REQUIRE(varRetained == Approx(1.0).epsilon(1e-7));

  data = origData;
  varRetained = p.Apply(data, 1.0);

  REQUIRE(data.n_rows == 3);
  REQUIRE(data.n_cols == 5);
  REQUIRE(varRetained == Approx(1.0).epsilon(1e-7));
}

/**
 * Compare the output of our exact PCA implementation with Armadillo's.
 */
TEST_CASE("ArmaComparisonExactPCATest", "[PCATest]")
{
  ArmaComparisonPCA<ExactSVDPolicy>();
}

/**
 * Compare the output of our randomized block krylov PCA implementation with
 * Armadillo's.
 */
TEST_CASE("ArmaComparisonRandomizedBlockKrylovPCATest", "[PCATest]")
{
  RandomizedBlockKrylovSVDPolicy decomposition(5);
  ArmaComparisonPCA<RandomizedBlockKrylovSVDPolicy>(false, decomposition);
}

/**
 * Compare the output of our randomized-SVD PCA implementation with Armadillo's.
 */
TEST_CASE("ArmaComparisonRandomizedPCATest", "[PCATest]")
{
  ArmaComparisonPCA<RandomizedSVDPCAPolicy>();
}

/**
 * Test that dimensionality reduction with exact-svd PCA works the same way
 * MATLAB does (which should be correct!).
 */
TEST_CASE("ExactPCADimensionalityReductionTest", "[PCATest]")
{
  PCADimensionalityReduction<ExactSVDPolicy>();
}

/**
 * Test that dimensionality reduction with randomized block krylov PCA works the
 * same way MATLAB does (which should be correct!).
 */
TEST_CASE("RandomizedBlockKrylovPCADimensionalityReductionTest", "[PCATest]")
{
  RandomizedBlockKrylovSVDPolicy decomposition(5);
  PCADimensionalityReduction<RandomizedBlockKrylovSVDPolicy>(false,
      decomposition);
}

/**
 * Test that dimensionality reduction with randomized-svd PCA works the same way
 * MATLAB does (which should be correct!).
 */
TEST_CASE("RandomizedPCADimensionalityReductionTest", "[PCATest]")
{
  PCADimensionalityReduction<RandomizedSVDPCAPolicy>();
}

/**
 * Test that dimensionality reduction with QUIC-SVD PCA works the same way
 * as the Exact-SVD PCA method.
 */
TEST_CASE("QUICPCADimensionalityReductionTest", "[PCATest]")
{
  arma::mat data, data1;
  if (!data::Load("test_data_3_1000.csv", data))
    FAIL("Cannot load dataset test_data_3_1000.csv");
  data1 = data;

  arma::mat backupData(data);

  // It isn't guaranteed that the QUIC-SVD will match with the exact SVD method,
  // starting with random samples. If this works 1 of 5 times, I'm fine with
  // that. All I want to know is that the QUIC-SVD method is able to solve the
  // task and is at least as good as the exact method (plus a little bit for
  // noise).
  size_t successes = 0;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    if (trial > 0)
    {
      data = backupData;
      data1 = backupData;
    }

    PCA<ExactSVDPolicy> exactPCA;
    const double varRetainedExact = exactPCA.Apply(data, 1);

    PCA<QUICSVDPolicy> quicPCA;
    const double varRetainedQUIC = quicPCA.Apply(data1, 1);

    if (std::abs(varRetainedExact - varRetainedQUIC) < 0.2)
    {
      ++successes;
      break;
    }
  }

  REQUIRE(successes >= 1);
  REQUIRE(data.n_rows == data1.n_rows);
  REQUIRE(data.n_cols == data1.n_cols);
}

/**
 * Test that setting the variance retained parameter to perform dimensionality
 * reduction works using the exact svd PCA method.
 */
TEST_CASE("ExactPCAVarianceRetainedTest", "[PCATest]")
{
  PCAVarianceRetained<ExactSVDPolicy>();
}

/**
 * Test that scaling PCA works.
 */
TEST_CASE("PCAScalingTest", "[PCATest]")
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
  PCA<> p(true);
  arma::mat transData;
  arma::vec eigval;
  arma::mat eigvec;

  p.Apply(data, transData, eigval, eigvec);

  // The first two components of the eigenvector with largest eigenvalue should
  // be somewhere near sqrt(2) / 2.  The third component should be close to
  // zero.  There is noise, of course...
  REQUIRE(std::abs(eigvec(0, 0)) == Approx(sqrt(2) / 2).epsilon(0.0035));
  REQUIRE(std::abs(eigvec(1, 0)) == Approx(sqrt(2) / 2).epsilon(0.0035));
  // Large tolerance for noise.
  REQUIRE(eigvec(2, 0) == Approx(0.0).margin(0.1));

  // The second component should be focused almost entirely in the third
  // dimension.
  REQUIRE(eigvec(0, 1) == Approx(0.0).margin(0.1));
  REQUIRE(eigvec(1, 1) == Approx(0.0).margin(0.1));
  REQUIRE(std::abs(eigvec(2, 1)) == Approx(1.0).epsilon(0.0035));

  // The third component should have the same absolute value characteristics as
  // the first (plus tolerance).
  REQUIRE(std::abs(eigvec(0, 0)) == Approx(sqrt(2) / 2).epsilon(0.0035));
  REQUIRE(std::abs(eigvec(1, 0)) == Approx(sqrt(2) / 2).epsilon(0.0035));
  // Large tolerance for noise.
  REQUIRE(eigvec(2, 0) == Approx(0.0).margin(0.1));

  // The eigenvalues should sum to three.
  REQUIRE(accu(eigval) == Approx(3.0).epsilon(0.001));
}

/**
 * Test PCA on a subview of a matrix with different decomposition strategies.
 */
TEMPLATE_TEST_CASE("PCASubviewTest", "[PCATest]", ExactSVDPolicy,
    RandomizedSVDPCAPolicy, RandomizedBlockKrylovSVDPolicy, QUICSVDPolicy)
{
  typedef TestType DecompositionPolicy;

  // Generate an artifical dataset in 10 dimensions.
  arma::mat data(3, 5000);

  arma::vec mean("1.0 3.0 -12.0");
  arma::mat cov("1.0 0.9 0.0;"
                "0.9 1.0 0.0;"
                "0.0 0.0 12.0");
  GaussianDistribution g(mean, cov);

  for (size_t i = 0; i < 5000; ++i)
    data.col(i) = g.Random();

  // Compute PCA on the first 2000 points.
  arma::mat transData1, transData2, transData3, eigvec;
  arma::vec eigval1, eigval2;

  PCA<DecompositionPolicy> p;
  p.Apply(data.cols(0, 1999), transData1);
  p.Apply(data.cols(0, 1999), transData2, eigval1);
  p.Apply(data.cols(0, 1999), transData3, eigval2, eigvec);

  // Only check for deterministic policies.
  if (std::is_same<DecompositionPolicy, ExactSVDPolicy>::value)
  {
    arma::mat trueTransData, trueEigvec;
    arma::vec trueEigval;

    arma::mat dataSub = data.cols(0, 1999);
    p.Apply(dataSub, trueTransData, trueEigval, trueEigvec);

    REQUIRE(arma::approx_equal(transData1, trueTransData, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(transData2, trueTransData, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(transData3, trueTransData, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(eigval1, trueEigval, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(eigval2, trueEigval, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(eigvec, trueEigvec, "both", 1e-5, 1e-5));
  }
}

/**
 * Test PCA on an input expression.
 */
TEMPLATE_TEST_CASE("PCAExpressionTest", "[PCATest]", ExactSVDPolicy,
    RandomizedSVDPCAPolicy, RandomizedBlockKrylovSVDPolicy, QUICSVDPolicy)
{
  typedef TestType DecompositionPolicy;

  // Generate an artifical dataset in 10 dimensions.
  arma::mat data(3, 5000);

  arma::vec mean("1.0 3.0 -12.0");
  arma::mat cov("1.0 0.9 0.0;"
                "0.9 1.0 0.0;"
                "0.0 0.0 12.0");
  GaussianDistribution g(mean, cov);

  for (size_t i = 0; i < 5000; ++i)
    data.col(i) = g.Random();

  // Compute PCA on an expression involving the input matrix.
  arma::mat transData1, transData2, transData3, eigvec;
  arma::vec eigval1, eigval2;

  PCA<DecompositionPolicy> p;
  p.Apply(2 * data + 1, transData1);
  p.Apply(2 * data + 1, transData2, eigval1);
  p.Apply(2 * data + 1, transData3, eigval2, eigvec);

  // Only check for deterministic policies.
  if (std::is_same<DecompositionPolicy, ExactSVDPolicy>::value)
  {
    arma::mat trueTransData, trueEigvec;
    arma::vec trueEigval;

    arma::mat dataSub = 2 * data + 1;
    p.Apply(dataSub, trueTransData, trueEigval, trueEigvec);

    REQUIRE(arma::approx_equal(transData1, trueTransData, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(transData2, trueTransData, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(transData3, trueTransData, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(eigval1, trueEigval, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(eigval2, trueEigval, "both", 1e-5, 1e-5));
    REQUIRE(arma::approx_equal(eigvec, trueEigvec, "both", 1e-5, 1e-5));
  }
}

/**
 * Test PCA on 32-bit data.
 */
TEMPLATE_TEST_CASE("PCAFloatTest", "[PCATest]", ExactSVDPolicy,
    RandomizedSVDPCAPolicy, RandomizedBlockKrylovSVDPolicy, QUICSVDPolicy)
{
  typedef TestType DecompositionPolicy;

  // Generate an artifical dataset in 10 dimensions.
  arma::fmat data(3, 5000);

  arma::vec mean("1.0 3.0 -12.0");
  arma::mat cov("1.0 0.9 0.0;"
                "0.9 1.0 0.0;"
                "0.0 0.0 12.0");
  GaussianDistribution g(mean, cov);

  for (size_t i = 0; i < 5000; ++i)
    data.col(i) = arma::conv_to<arma::fvec>::from(g.Random());

  // Compute PCA on the floating-point data.
  arma::fmat coeff, coeff1, score, score1;
  arma::fvec eigVal, eigVal1;

  PCA<DecompositionPolicy> p;
  p.Apply(data, score1, eigVal1, coeff1);

  princomp(coeff, score, eigVal, trans(data));

  // Verify the PCA results based on the eigenvalues.  We don't check for
  // QUIC-SVD, since that method has a lot of noise.
  if (!std::is_same<DecompositionPolicy, QUICSVDPolicy>::value)
  {
    for (size_t i = 0; i < eigVal.n_elem; ++i)
    {
      if (eigVal[i] == 0.0)
        REQUIRE(eigVal1[i] == Approx(0.0).margin(1e-5));
      else
        REQUIRE(eigVal[i] == Approx(eigVal1[i]).epsilon(1e-3));
    }
  }
}

/**
 * Test that we can convert sparse input matrices to dense output matrices.
 * We check the general shape of the output, but not the exact details---those
 * are handled in other tests.
 */
TEMPLATE_TEST_CASE("PCASparseToDenseTest", "[PCATest]", float, double)
{
  typedef arma::Mat<TestType> MatType;
  typedef arma::SpMat<TestType> SpMatType;

  SpMatType dataset;
  dataset.sprandu(1000, 50000, 0.01);
  MatType transformedDataset1, transformedDataset2;

  PCA<> p;
  const double varRetained1 = p.Apply(dataset, transformedDataset1, 5);
  const double varRetained2 = p.Apply(dataset, transformedDataset2, 0.6);

  REQUIRE(transformedDataset1.n_cols == dataset.n_cols);
  REQUIRE(transformedDataset2.n_cols == dataset.n_cols);
  REQUIRE(varRetained1 >= 0.0);
  REQUIRE(varRetained1 <= 1.0);
  REQUIRE(varRetained2 >= 0.0);
  REQUIRE(varRetained2 <= 1.0);
  REQUIRE(transformedDataset1.n_rows == 5);
  REQUIRE(transformedDataset2.n_rows <= dataset.n_rows);

  // Ensure we get basically the same as if we had done it to a dense matrix.
  MatType denseData1(dataset);
  MatType denseData2(dataset);

  const double varRetained3 = p.Apply(denseData1, 5);
  const double varRetained4 = p.Apply(denseData2, 0.6);

  REQUIRE(varRetained1 == Approx(varRetained3));
  REQUIRE(varRetained2 == Approx(varRetained4));

  REQUIRE(denseData2.n_rows == transformedDataset2.n_rows);
}
