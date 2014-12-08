/**
 * @file quic_svd_test.cpp
 * @author Siddharth Agrawal
 *
 * Test file for QUIC-SVD class.
 *
 * This file is part of MLPACK 1.0.11.
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

  // Since QUIC-SVD may have random errors, run up to three trials to get a good
  // results.
  size_t successes = 0;
  size_t trial = 0;

  while (trial < 3 && successes < 1)
  {
    // Obtain the SVD using default parameters.
    arma::mat u, v, sigma;
    QUIC_SVD quicsvd(dataset, u, v, sigma);

    // Reconstruct the matrix using the SVD.
    arma::mat reconstruct;
    reconstruct = u * sigma * v.t();

    // The relative reconstruction error should be small.
    double relativeError = arma::norm(dataset - reconstruct, "frob") /
                           arma::norm(dataset, "frob");
    if (relativeError < 1e-5)
      ++successes;

    ++trial;
  }

  BOOST_REQUIRE_GE(successes, 1);
}

BOOST_AUTO_TEST_SUITE_END();
