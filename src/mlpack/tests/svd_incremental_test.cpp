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
TEST_CASE("SVDIncompleteIncrementalConvergenceTest", "[SVDIncrementalTest]")
{
  sp_mat data;
  data.sprandn(100, 100, 0.2);

  SVDIncompleteIncrementalLearning svd(0.01);
  IncompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> > iit;

  AMF<IncompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> >,
      RandomAMFInitialization,
      SVDIncompleteIncrementalLearning> amf(
      iit, RandomAMFInitialization(), svd);

  mat m1, m2;
  amf.Apply(data, 2, m1, m2);

  REQUIRE(amf.TerminationPolicy().Iteration() !=
          amf.TerminationPolicy().MaxIterations());
}

/**
 * Test for convergence of complete incremenal learning
 */
TEST_CASE("SVDCompleteIncrementalConvergenceTest", "[SVDIncrementalTest]")
{
  sp_mat data;
  data.sprandn(100, 100, 0.2);

  SVDCompleteIncrementalLearning<sp_mat> svd(0.01);
  CompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> > iit;

  AMF<CompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> >,
      RandomAMFInitialization,
      SVDCompleteIncrementalLearning<sp_mat> > amf(iit,
                                                   RandomAMFInitialization(),
                                                   svd);
  mat m1, m2;
  amf.Apply(data, 2, m1, m2);

  REQUIRE(amf.TerminationPolicy().Iteration() !=
          amf.TerminationPolicy().MaxIterations());
}

//! This is used to ensure we start from the same initial point.
class SpecificRandomInitialization
{
 public:
  SpecificRandomInitialization(const size_t n, const size_t r, const size_t m) :
      W(arma::randu<arma::mat>(n, r)),
      H(arma::randu<arma::mat>(r, m)) { }

  template<typename MatType>
  inline void Initialize(const MatType& /* V */,
                         const size_t /* r */,
                         arma::mat& W,
                         arma::mat& H)
  {
    W = this->W;
    H = this->H;
  }

 private:
  arma::mat W;
  arma::mat H;
};

TEST_CASE("SVDIncompleteIncrementalRegularizationTest", "[SVDIncrementalTest]")
{
  mat dataset;
  if (!data::Load("GroupLensSmall.csv", dataset))
    FAIL("Cannot load dataset GroupLensSmall.csv");

  // Generate list of locations for batch insert constructor for sparse
  // matrices.
  arma::umat locations(2, dataset.n_cols);
  arma::vec values(dataset.n_cols);
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
  sp_mat cleanedData = arma::sp_mat(locations, values, maxUserID, maxItemID);
  sp_mat cleanedData2 = cleanedData;

  SpecificRandomInitialization sri(cleanedData.n_rows, 2, cleanedData.n_cols);

  ValidationRMSETermination<sp_mat> vrt(cleanedData, 2000);
  AMF<IncompleteIncrementalTermination<ValidationRMSETermination<sp_mat> >,
      SpecificRandomInitialization,
      SVDIncompleteIncrementalLearning> amf1(vrt, sri,
      SVDIncompleteIncrementalLearning(0.001, 0, 0));

  mat m1, m2;
  double regularRMSE = amf1.Apply(cleanedData, 2, m1, m2);

  ValidationRMSETermination<sp_mat> vrt2(cleanedData2, 2000);
  AMF<IncompleteIncrementalTermination<ValidationRMSETermination<sp_mat> >,
      SpecificRandomInitialization,
      SVDIncompleteIncrementalLearning> amf2(vrt2, sri,
      SVDIncompleteIncrementalLearning(0.001, 0.01, 0.01));

  mat m3, m4;
  double regularizedRMSE = amf2.Apply(cleanedData2, 2, m3, m4);

  REQUIRE(regularizedRMSE < regularRMSE + 0.105);
}
