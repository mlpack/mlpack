/**
 * @file test_tools.hpp
 * @author Ryan Curtin
 *
 * This file includes some useful macros for tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_TEST_TOOLS_HPP
#define MLPACK_TESTS_TEST_TOOLS_HPP

#include <mlpack/core.hpp>
#include <boost/version.hpp>

// Require the approximation L to be within a relative error of E respect to the
// actual value R.
#define REQUIRE_RELATIVE_ERR(L, R, E) \
    BOOST_REQUIRE_LE(std::abs((R) - (L)), (E) * std::abs(R))

// Check the values of two matrices.
inline void CheckMatrices(const arma::mat& a,
                          const arma::mat& b,
                          double tolerance = 1e-5)
{
  BOOST_REQUIRE_EQUAL(a.n_rows, b.n_rows);
  BOOST_REQUIRE_EQUAL(a.n_cols, b.n_cols);

  for (size_t i = 0; i < a.n_elem; ++i)
  {
    if (std::abs(a[i]) < tolerance / 2)
      BOOST_REQUIRE_SMALL(b[i], tolerance / 2);
    else
      BOOST_REQUIRE_CLOSE(a[i], b[i], tolerance);
  }
}

// Check the values of two unsigned matrices.
inline void CheckMatrices(const arma::Mat<size_t>& a,
                          const arma::Mat<size_t>& b)
{
  BOOST_REQUIRE_EQUAL(a.n_rows, b.n_rows);
  BOOST_REQUIRE_EQUAL(a.n_cols, b.n_cols);

  for (size_t i = 0; i < a.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(a[i], b[i]);
}

// Check the values of two cubes.
inline void CheckMatrices(const arma::cube& a,
                          const arma::cube& b,
                          double tolerance = 1e-5)
{
  BOOST_REQUIRE_EQUAL(a.n_rows, b.n_rows);
  BOOST_REQUIRE_EQUAL(a.n_cols, b.n_cols);
  BOOST_REQUIRE_EQUAL(a.n_slices, b.n_slices);

  for (size_t i = 0; i < a.n_elem; ++i)
  {
    if (std::abs(a[i]) < tolerance / 2)
      BOOST_REQUIRE_SMALL(b[i], tolerance / 2);
    else
      BOOST_REQUIRE_CLOSE(a[i], b[i], tolerance);
  }
}

// Check if two matrices are different.
inline void CheckMatricesNotEqual(const arma::mat& a,
                                  const arma::mat& b,
                                  double tolerance = 1e-5)
{
  bool areDifferent = false;

  // Only check the elements if the dimensions are equal.
  if (a.n_rows == b.n_rows && a.n_cols == b.n_cols)
  {
    for (size_t i = 0; i < a.n_elem; ++i)
    {
      if (std::abs(a[i]) < tolerance / 2 &&
          b[i] > tolerance / 2)
      {
        areDifferent = true;
        break;
      }
      else if (std::abs(a[i] - b[i]) > tolerance)
      {
        areDifferent = true;
        break;
      }
    }
  }
  else
    areDifferent = true;

  if (!areDifferent)
    BOOST_ERROR("The matrices are equal.");
}

// Check if two unsigned matrices are different.
inline void CheckMatricesNotEqual(const arma::Mat<size_t>& a,
                                  const arma::Mat<size_t>& b)
{
  bool areDifferent = false;

  // Only check the elements if the dimensions are equal.
  if (a.n_rows == b.n_rows && a.n_cols == b.n_cols)
  {
    for (size_t i = 0; i < a.n_elem; ++i)
    {
      if (a[i] != b[i])
      {
        areDifferent = true;
        break;
      }
    }
  }
  else
    areDifferent = true;

  if (!areDifferent)
    BOOST_ERROR("The matrices are equal.");
}

// Check if two cubes are different.
inline void CheckMatricesNotEqual(const arma::cube& a,
                                  const arma::cube& b,
                                  double tolerance = 1e-5)
{
  bool areDifferent = false;

  // Only check the elements if the dimensions are equal.
  if (a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
      a.n_slices == b.n_slices)
  {
    for (size_t i = 0; i < a.n_elem; ++i)
    {
      if (std::abs(a[i]) < tolerance / 2 &&
          b[i] > tolerance / 2)
      {
        areDifferent = true;
        break;
      }
      else if (std::abs(a[i] - b[i]) > tolerance)
      {
        areDifferent = true;
        break;
      }
    }
  }
  else
    areDifferent = true;

  if (!areDifferent)
    BOOST_ERROR("The matrices are equal.");
}

// Filter typeinfo string to generate unique filenames for serialization tests.
inline std::string FilterFileName(const std::string& inputString)
{
  // Take the last valid 32 characters for the filename.
  std::string fileName;
  for (auto it = inputString.rbegin(); it != inputString.rend() &&
      fileName.size() != 32; ++it)
  {
    if (std::isalnum(*it))
      fileName.push_back(*it);
  }

  return fileName;
}

#endif
