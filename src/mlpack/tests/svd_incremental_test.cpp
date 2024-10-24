/**
 * @file tests/svd_incremental_test.cpp
 * @author Sumedh Ghaisas
 *
 * Tests for SVDIncompleteIncrementalLearning and
 * SVDCompleteIncrementalLearning.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/amf.hpp>

#include "catch.hpp"

using namespace std;
using namespace mlpack;
using namespace arma;

/**
 * Test for convergence of incomplete incremenal learning.
 */
TEMPLATE_TEST_CASE("SVDIncompleteIncrementalConvergenceTest",
    "[SVDIncrementalTest]", float, double)
{
  using eT = TestType;

  SpMat<eT> data;
  data.sprandn(100, 100, 0.2);

  SVDIncompleteIncrementalLearning<SpMat<eT>> svd(0.01);
  IncompleteIncrementalTermination<
      SimpleToleranceTermination<SpMat<eT>, Mat<eT>>> iit;

  AMF<IncompleteIncrementalTermination<
          SimpleToleranceTermination<SpMat<eT>, Mat<eT>>>,
      RandomAMFInitialization,
      SVDIncompleteIncrementalLearning<SpMat<eT>>> amf(
      iit, RandomAMFInitialization(), svd);

  Mat<eT> m1, m2;
  amf.Apply(data, 2, m1, m2);

  REQUIRE(amf.TerminationPolicy().Iteration() !=
          amf.TerminationPolicy().MaxIterations());
}

/**
 * Test for convergence of complete incremenal learning
 */
TEMPLATE_TEST_CASE("SVDCompleteIncrementalConvergenceTest",
    "[SVDIncrementalTest]", float, double)
{
  using eT = TestType;

  SpMat<eT> data;
  data.sprandn(100, 100, 0.2);

  SVDCompleteIncrementalLearning<SpMat<eT>> svd(0.01);
  CompleteIncrementalTermination<SimpleToleranceTermination<SpMat<eT>, Mat<eT>>>
      iit;

  AMF<CompleteIncrementalTermination<
          SimpleToleranceTermination<SpMat<eT>, Mat<eT>>>,
      RandomAMFInitialization,
      SVDCompleteIncrementalLearning<SpMat<eT>>>
      amf(iit, RandomAMFInitialization(), svd);
  Mat<eT> m1, m2;
  amf.Apply(data, 2, m1, m2);

  REQUIRE(amf.TerminationPolicy().Iteration() !=
          amf.TerminationPolicy().MaxIterations());
}

//! This is used to ensure we start from the same initial point.
template<typename MatType>
class SpecificRandomInitialization
{
 public:
  SpecificRandomInitialization(const size_t n, const size_t r, const size_t m) :
      W(arma::randu<MatType>(n, r)),
      H(arma::randu<MatType>(r, m)) { }

  template<typename VMatType>
  inline void Initialize(const VMatType& /* V */,
                         const size_t /* r */,
                         MatType& W,
                         MatType& H)
  {
    W = this->W;
    H = this->H;
  }

 private:
  MatType W;
  MatType H;
};

TEMPLATE_TEST_CASE("SVDIncompleteIncrementalRegularizationTest",
    "[SVDIncrementalTest]", float, double)
{
  using eT = TestType;

  Mat<eT> dataset;
  if (!data::Load("GroupLensSmall.csv", dataset))
    FAIL("Cannot load dataset GroupLensSmall.csv");

  // Generate list of locations for batch insert constructor for sparse
  // matrices.
  arma::umat locations(2, dataset.n_cols);
  arma::Col<eT> values(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    // We have to transpose it because items are rows, and users are columns.
    locations(0, i) = ((arma::uword) dataset(0, i));
    locations(1, i) = ((arma::uword) dataset(1, i));
    values(i) = dataset(2, i);
  }

  // Find maximum user and item IDs.
  const size_t maxUserID = (size_t) max(locations.row(0)) + 1;
  const size_t maxItemID = (size_t) max(locations.row(1)) + 1;

  // Fill sparse matrix.
  SpMat<eT> cleanedData(locations, values, maxUserID, maxItemID);
  SpMat<eT> cleanedData2 = cleanedData;

  SpecificRandomInitialization<Mat<eT>> sri(cleanedData.n_rows, 2,
      cleanedData.n_cols);

  ValidationRMSETermination<SpMat<eT>, Mat<eT>> vrt(cleanedData, 2000);
  AMF<IncompleteIncrementalTermination<
          ValidationRMSETermination<SpMat<eT>, Mat<eT>>>,
      SpecificRandomInitialization<Mat<eT>>,
      SVDIncompleteIncrementalLearning<SpMat<eT>>> amf1(vrt, sri,
      SVDIncompleteIncrementalLearning<SpMat<eT>>(0.001, 0, 0));

  Mat<eT> m1, m2;
  double regularRMSE = amf1.Apply(cleanedData, 2, m1, m2);

  ValidationRMSETermination<SpMat<eT>, Mat<eT>> vrt2(cleanedData2, 2000);
  AMF<IncompleteIncrementalTermination<
          ValidationRMSETermination<SpMat<eT>, Mat<eT>>>,
      SpecificRandomInitialization<Mat<eT>>,
      SVDIncompleteIncrementalLearning<SpMat<eT>>> amf2(vrt2, sri,
      SVDIncompleteIncrementalLearning<SpMat<eT>>(0.001, 0.01, 0.01));

  Mat<eT> m3, m4;
  double regularizedRMSE = amf2.Apply(cleanedData2, 2, m3, m4);

  REQUIRE(regularizedRMSE < regularRMSE + 0.105);
}
