/**
 * @file tests/block_krylov_svd_test.cpp
 * @author Marcus Edel
 *
 * Test file for the Randomized Block Krylov SVD class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/block_krylov_svd.hpp>

#include "catch.hpp"

using namespace mlpack;

// Generate a low rank matrix with bell-shaped singular values.
void CreateNoisyLowRankMatrix(arma::mat& data,
                              const size_t rows,
                              const size_t cols,
                              const size_t rank,
                              const double strength)
{
  arma::mat R, U, V;
  const size_t n = std::min(rows, cols);

  arma::qr_econ(U, R, arma::randn<arma::mat>(rows, n));
  arma::qr_econ(V, R, arma::randn<arma::mat>(cols, n));

  arma::vec ids = arma::linspace<arma::vec>(0, n - 1, n);

  arma::vec lowRank = ((1 - strength) *
      arma::exp(-1.0 * arma::pow((ids / rank), 2)));
  arma::vec tail = strength * arma::exp(-0.1 * ids / rank);

  arma::mat s = arma::zeros<arma::mat>(n, n);
  s.diag() = lowRank + tail;
  data = (U * s) * V.t();
}

/**
 * The reconstruction and sigular value error of the obtained SVD should be
 * small.
 */
TEST_CASE("RandomizedBlockKrylovSVDReconstructionError",
          "[BlockKrylovSVDTest]")
{
  arma::mat U = arma::randn<arma::mat>(3, 20);
  arma::mat V = arma::randn<arma::mat>(10, 3);

  arma::mat R;
  arma::qr_econ(U, R, U);
  arma::qr_econ(V, R, V);

  arma::mat s = arma::diagmat(arma::vec("1 0.1 0.01"));

  arma::mat data = arma::trans(U * arma::diagmat(s) * V.t());

  // Center the data into a temporary matrix.
  arma::mat centeredData;
  Center(data, centeredData);

  arma::mat U1, U2, V1, V2;
  arma::vec s1, s2, s3;

  arma::svd_econ(U1, s1, V1, centeredData);

  RandomizedBlockKrylovSVD rSVD(20, 10);
  rSVD.Apply(centeredData, U2, s2, V2, 3);

  // Use the same amount of data for the compariosn (matrix rank).
  s3 = s1.subvec(0, s2.n_elem - 1);

  // The sigular value error should be small.
  double error = arma::norm(s2 - s3, "frob") / arma::norm(s2, "frob");
  REQUIRE(error == Approx(0.0).margin(1e-5));

  arma::mat reconstruct = U2 * arma::diagmat(s2) * V2.t();

  // The relative reconstruction error should be small.
  error = arma::norm(centeredData - reconstruct, "frob") /
      arma::norm(centeredData, "frob");
  REQUIRE(error == Approx(0.0).margin(1e-7));
}

/*
 * Check if the method can handle noisy matrices.
 */
TEST_CASE("RandomizedBlockKrylovSVDNoisyLowRankTest", "[BlockKrylovSVDTest]")
{
  arma::mat data;
  CreateNoisyLowRankMatrix(data, 200, 1000, 5, 0.5);

  const size_t rank = 5;

  arma::mat U1, U2, V1, V2;
  arma::vec s1, s2, s3;

  arma::svd_econ(U1, s1, V1, data);

  RandomizedBlockKrylovSVD rSVDB(data, U2, s2, V2, 10, rank, 20);

  double error = arma::max(arma::abs(s1.subvec(0, rank) - s2.subvec(0, rank)));
  REQUIRE(error == Approx(0.0).margin(1e-4));
}
