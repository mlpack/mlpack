/**
 * @file quic_svd_test.cpp
 * @author Siddharth Agrawal
 *
 * Test file for QUIC-SVD class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/quic_svd/quic_svd.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

BOOST_AUTO_TEST_SUITE(QUICSVDTest);

using namespace mlpack;

/**
 * The reconstruction error of the obtained SVD should be small.
 */
BOOST_AUTO_TEST_CASE(QUICSVDReconstructionError)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("test_data_3_1000.csv", dataset);

  // Obtain the SVD using default parameters.
  arma::mat u, v, sigma;
  svd::QUIC_SVD quicsvd(dataset, u, v, sigma);

  // Reconstruct the matrix using the SVD.
  arma::mat reconstruct;
  reconstruct = u * sigma * v.t();

  // The relative reconstruction error should be small.
  double relativeError = arma::norm(dataset - reconstruct, "frob") /
                         arma::norm(dataset, "frob");
  BOOST_REQUIRE_SMALL(relativeError, 1e-5);
}

/**
 * The singular value error of the obtained SVD should be small.
 */
BOOST_AUTO_TEST_CASE(QUICSVDSingularValueError)
{
  arma::mat U = arma::randn<arma::mat>(3, 20);
  arma::mat V = arma::randn<arma::mat>(10, 3);

  arma::mat R;
  arma::qr_econ(U, R, U);
  arma::qr_econ(V, R, V);

  arma::mat s = arma::diagmat(arma::vec("1 0.1 0.01"));

  arma::mat data = arma::trans(U * arma::diagmat(s) * V.t());

  arma::vec s1, s3;
  arma::mat U1, U2, V1, V2, s2;

  // Obtain the SVD using default parameters.
  arma::svd_econ(U1, s1, V1, data);
  svd::QUIC_SVD quicsvd(data, U1, V1, s2);

  s3 = arma::diagvec(s2);
  s1 = s1.subvec(0, s3.n_elem - 1);

  // The sigular value error should be small.
  double error = arma::norm(s1 - s3);
  BOOST_REQUIRE_SMALL(error, 0.05);
}

BOOST_AUTO_TEST_CASE(QUICSVDSameDimensionTest)
{
  arma::mat dataset = arma::randn<arma::mat>(10, 10);

  // Obtain the SVD using default parameters.
  arma::mat u, v, sigma;
  svd::QUIC_SVD quicsvd(dataset, u, v, sigma);
}

BOOST_AUTO_TEST_SUITE_END();
