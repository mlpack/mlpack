/**
 * @file quic_svd_test.cpp
 * @author Siddharth Agrawal
 *
 * Test file for QUIC-SVD class.
 *
 * This file is part of mlpack 2.0.1.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/quic_svd/quic_svd.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(QUICSVDTest);

using namespace mlpack;
using namespace mlpack::svd;

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
  QUIC_SVD quicsvd(dataset, u, v, sigma);

  // Reconstruct the matrix using the SVD.
  arma::mat reconstruct;
  reconstruct = u * sigma * v.t();

  // The relative reconstruction error should be small.
  double relativeError = arma::norm(dataset - reconstruct, "frob") /
                         arma::norm(dataset, "frob");
  BOOST_REQUIRE_SMALL(relativeError, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
