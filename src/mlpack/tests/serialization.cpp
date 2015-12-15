/**
 * @file serialization.cpp
 * @author Ryan Curtin
 *
 * Miscellaneous utility functions for serialization tests.
 */
#include "serialization.hpp"

namespace mlpack {

// Utility function to check the equality of two Armadillo matrices.
void CheckMatrices(const arma::mat& x,
                   const arma::mat& xmlX,
                   const arma::mat& textX,
                   const arma::mat& binaryX)
{
  // First check dimensions.
  BOOST_REQUIRE_EQUAL(x.n_rows, xmlX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, textX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, binaryX.n_rows);

  BOOST_REQUIRE_EQUAL(x.n_cols, xmlX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, textX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, binaryX.n_cols);

  BOOST_REQUIRE_EQUAL(x.n_elem, xmlX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, textX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, binaryX.n_elem);

  // Now check elements.
  for (size_t i = 0; i < x.n_elem; ++i)
  {
    const double val = x[i];
    if (val == 0.0)
    {
      BOOST_REQUIRE_SMALL(xmlX[i], 1e-8);
      BOOST_REQUIRE_SMALL(textX[i], 1e-8);
      BOOST_REQUIRE_SMALL(binaryX[i], 1e-8);
    }
    else
    {
      BOOST_REQUIRE_CLOSE(val, xmlX[i], 1e-8);
      BOOST_REQUIRE_CLOSE(val, textX[i], 1e-8);
      BOOST_REQUIRE_CLOSE(val, binaryX[i], 1e-8);
    }
  }
}

void CheckMatrices(const arma::Mat<size_t>& x,
                   const arma::Mat<size_t>& xmlX,
                   const arma::Mat<size_t>& textX,
                   const arma::Mat<size_t>& binaryX)
{
  // First check dimensions.
  BOOST_REQUIRE_EQUAL(x.n_rows, xmlX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, textX.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_rows, binaryX.n_rows);

  BOOST_REQUIRE_EQUAL(x.n_cols, xmlX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, textX.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_cols, binaryX.n_cols);

  BOOST_REQUIRE_EQUAL(x.n_elem, xmlX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, textX.n_elem);
  BOOST_REQUIRE_EQUAL(x.n_elem, binaryX.n_elem);

  // Now check elements.
  for (size_t i = 0; i < x.n_elem; ++i)
  {
    BOOST_REQUIRE_EQUAL(x[i], xmlX[i]);
    BOOST_REQUIRE_EQUAL(x[i], textX[i]);
    BOOST_REQUIRE_EQUAL(x[i], binaryX[i]);
  }
}

} // namespace mlpack
