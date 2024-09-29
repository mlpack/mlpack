/**
 * @file tests/randomized_svd_test.cpp
 * @author Marcus Edel
 *
 * Test file for the Randomized SVD class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/randomized_svd.hpp>

#include "catch.hpp"

using namespace mlpack;

/**
 * The reconstruction and sigular value error of the obtained SVD should be
 * small.
 */
TEST_CASE("RandomizedSVDReconstructionError", "[RandomizedSVDTest]")
{
  arma::mat U = arma::randn<arma::mat>(3, 20);
  arma::mat V = arma::randn<arma::mat>(10, 3);

  arma::mat R;
  arma::qr_econ(U, R, U);
  arma::qr_econ(V, R, V);

  arma::mat s = arma::diagmat(arma::vec("1 0.1 0.01"));

  arma::mat data = trans(U * arma::diagmat(s) * V.t());

  // Center the data into a temporary matrix.
  arma::mat centeredData = data.each_col() - arma::mean(data, 1);

  arma::mat U1, U2, V1, V2;
  arma::vec s1, s2, s3;

  arma::svd_econ(U1, s1, V1, centeredData);

  RandomizedSVD rSVD(0, 10);
  rSVD.Apply(data, U2, s2, V2, 3);

  // Use the same amount of data for the compariosn (matrix rank).
  s3 = s1.subvec(0, s2.n_elem - 1);

  // The sigular value error should be small.
  double error = arma::norm(s2 - s3, "frob") / arma::norm(s2, "frob");
  REQUIRE(error == Approx(0.0).margin(1e-5));

  arma::mat reconstruct = U2 * arma::diagmat(s2) * V2.t();

  // The relative reconstruction error should be small.
  error = arma::norm(centeredData - reconstruct, "frob") /
      arma::norm(centeredData, "frob");
  REQUIRE(error == Approx(0.0).margin(1e-5));
}
