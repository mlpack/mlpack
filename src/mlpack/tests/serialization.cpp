/**
 * @file tests/serialization.cpp
 * @author Ryan Curtin
 *
 * Miscellaneous utility functions for serialization tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "serialization.hpp"

namespace mlpack {

// Utility function to check the equality of two Armadillo matrices.
void CheckMatrices(const arma::mat& x,
                   const arma::mat& xmlX,
                   const arma::mat& jsonX,
                   const arma::mat& binaryX)
{
  // First check dimensions.
  BOOST_REQUIRE_EQUAL(x.n_rows, xmlX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, jsonX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, binaryX.n_rows);

  BOOST_REQUIRE_EQUAL(x.n_cols, xmlX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, jsonX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, binaryX.n_cols);

  BOOST_REQUIRE_EQUAL(x.n_elem, xmlX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, jsonX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, binaryX.n_elem);

  // Now check elements.
  for (size_t i = 0; i < x.n_elem; ++i)
  {
    const double val = x[i];
    if (val == 0.0)
    {
      BOOST_REQUIRE_SMALL(xmlX[i], 1e-6);
      BOOST_REQUIRE_SMALL(jsonX[i], 1e-6);
      BOOST_REQUIRE_SMALL(binaryX[i], 1e-6);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(val, xmlX[i], 1e-6);
      BOOST_REQUIRE_CLOSE(val, jsonX[i], 1e-6);
      BOOST_REQUIRE_CLOSE(val, binaryX[i], 1e-6);
    }
  }
}

void CheckMatrices(const arma::Mat<size_t>& x,
                   const arma::Mat<size_t>& xmlX,
                   const arma::Mat<size_t>& jsonX,
                   const arma::Mat<size_t>& binaryX)
{
  // First check dimensions.
  BOOST_REQUIRE_EQUAL(x.n_rows, xmlX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, jsonX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, binaryX.n_rows);

  BOOST_REQUIRE_EQUAL(x.n_cols, xmlX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, jsonX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, binaryX.n_cols);

  BOOST_REQUIRE_EQUAL(x.n_elem, xmlX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, jsonX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, binaryX.n_elem);

  // Now check elements.
  for (size_t i = 0; i < x.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(x[i], xmlX[i]);
    BOOST_REQUIRE_EQUAL(x[i], jsonX[i]);
    BOOST_REQUIRE_EQUAL(x[i], binaryX[i]);
  }
}

void CheckMatrices(const arma::cube& x,
                   const arma::cube& xmlX,
                   const arma::cube& jsonX,
                   const arma::cube& binaryX)
{
  // First check dimensions.
  BOOST_REQUIRE_EQUAL(x.n_rows, xmlX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, jsonX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, binaryX.n_rows);

  BOOST_REQUIRE_EQUAL(x.n_cols, xmlX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, jsonX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, binaryX.n_cols);

  BOOST_REQUIRE_EQUAL(x.n_slices, xmlX.n_slices);
  BOOST_REQUIRE_EQUAL(x.n_slices, jsonX.n_slices);
  BOOST_REQUIRE_EQUAL(x.n_slices, binaryX.n_slices);

  BOOST_REQUIRE_EQUAL(x.n_elem, xmlX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, jsonX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, binaryX.n_elem);

  // Now check elements.
  for (size_t i = 0; i < x.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(x[i], xmlX[i]);
    BOOST_REQUIRE_EQUAL(x[i], jsonX[i]);
    BOOST_REQUIRE_EQUAL(x[i], binaryX[i]);
  }
}

} // namespace mlpack
