/**
 * @file tests/test_catch_tools.hpp
 * @author Ryan Curtin
 *
 * This file includes some useful macros for tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_TEST_CATCH_TOOLS_HPP
#define MLPACK_TESTS_TEST_CATCH_TOOLS_HPP

#include <mlpack/core.hpp>

#include "catch.hpp"

// Require the approximation L to be within a relative error of E respect to the
// actual value R.
#define REQUIRE_RELATIVE_ERR(L, R, E) \
    REQUIRE(std::abs((R) - (L)) <= (E) * std::abs(R))

// Simple wrapper class to prevent copies of Armadillo matrices.
template <typename MatType, typename ElemType = typename MatType::elem_type,
          typename = std::enable_if_t<std::is_same_v<ElemType,
              typename MatType::elem_type>>>
class MatProxy
{
 public:
  explicit MatProxy(const MatType& mat) : mat(mat) { }

  const arma::Mat<ElemType> mat;
  arma::uword n_elem() const { return mat.n_elem; }
  arma::uword n_rows() const { return mat.n_rows; }
  arma::uword n_cols() const { return mat.n_cols; }
  ElemType operator[](const size_t i) const { return mat[i]; }
};

template <typename ElemType>
class MatProxy<arma::Mat<ElemType>, ElemType>
{
 public:
  explicit MatProxy(const arma::Mat<ElemType>& mat) : mat(mat) { }

  const arma::Mat<ElemType>& mat;
  arma::uword n_elem() const { return mat.n_elem; }
  arma::uword n_rows() const { return mat.n_rows; }
  arma::uword n_cols() const { return mat.n_cols; }
  ElemType operator[](const size_t i) const { return mat[i]; }
};

// Check the values of two matrices.
template <typename MatTypeA, typename MatTypeB,
    typename = std::enable_if_t<arma::is_arma_type<MatTypeA>::value
        && arma::is_arma_type<MatTypeB>::value
        && std::is_same_v<typename MatTypeA::elem_type,
                          typename MatTypeB::elem_type>
        && !std::is_integral_v<typename MatTypeA::elem_type>>>
inline void CheckMatrices(const MatTypeA& _a,
                          const MatTypeB& _b,
                          double tolerance = 1e-5)
{
  MatProxy<MatTypeA> a(_a);
  MatProxy<MatTypeB> b(_b);
  REQUIRE(a.n_rows() == b.n_rows());
  REQUIRE(a.n_cols() == b.n_cols());

  for (size_t i = 0; i < a.n_elem(); ++i)
  {
    if (std::abs(a[i]) < tolerance / 2)
      REQUIRE(b[i] == Approx(0.0).margin(tolerance / 2));
    else
      REQUIRE(a[i] == Approx(b[i]).epsilon(tolerance / 100));
  }
}

// Check the values of two unsigned matrices.
inline void CheckMatrices(const arma::Mat<size_t>& a,
                          const arma::Mat<size_t>& b)
{
  REQUIRE(a.n_rows == b.n_rows);
  REQUIRE(a.n_cols == b.n_cols);

  for (size_t i = 0; i < a.n_elem; ++i)
    REQUIRE(a[i] == b[i]);
}

template <typename FieldType,
          typename = std::enable_if_t<
              arma::is_arma_type<typename FieldType::object_type>::value>>
// Check the values of two field types.
inline void CheckFields(const FieldType& a,
                        const FieldType& b)
{
  REQUIRE(a.n_rows == b.n_rows);
  REQUIRE(a.n_cols == b.n_cols);

  for (size_t i = 0; i < a.n_slices; ++i)
    CheckMatrices(a(i), b(i));
}


// Simple wrapper class to prevent copies of Armadillo cubes.
template <typename CubeType, typename ElemType = typename CubeType::elem_type,
    typename = std::enable_if_t<std::is_same_v<ElemType,
        typename CubeType::elem_type>>>
class CubeProxy
{
 public:
  explicit CubeProxy(const CubeType& cube) : cube(cube) { }

  const arma::Cube<ElemType> cube;
  arma::uword n_elem() const { return cube.n_elem; }
  arma::uword n_rows() const { return cube.n_rows; }
  arma::uword n_cols() const { return cube.n_cols; }
  arma::uword n_slices() const { return cube.n_slices; }
  ElemType operator[](const size_t i) const { return cube[i]; }
};

template <typename ElemType>
class CubeProxy<arma::Cube<ElemType>, ElemType>
{
 public:
  explicit CubeProxy(const arma::Cube<ElemType>& cube) : cube(cube) { }

  const arma::Cube<ElemType>& cube;
  arma::uword n_elem() const { return cube.n_elem; }
  arma::uword n_rows() const { return cube.n_rows; }
  arma::uword n_cols() const { return cube.n_cols; }
  arma::uword n_slices() const { return cube.n_slices; }
  ElemType operator[](const size_t i) const { return cube[i]; }
};

// Check the values of two cubes.
template <typename CubeTypeA, typename CubeTypeB,
    typename = std::enable_if_t<arma::is_arma_cube_type<CubeTypeA>::value
        && arma::is_arma_cube_type<CubeTypeB>::value
        && std::is_same_v<typename CubeTypeA::elem_type,
                          typename CubeTypeB::elem_type>
        && !std::is_integral_v<typename CubeTypeA::elem_type>>,
    typename = void>
inline void CheckMatrices(const CubeTypeA& _a,
                          const CubeTypeB& _b,
                          double tolerance = 1e-5)
{
  CubeProxy<CubeTypeA> a(_a);
  CubeProxy<CubeTypeB> b(_b);
  REQUIRE(a.n_rows() == b.n_rows());
  REQUIRE(a.n_cols() == b.n_cols());
  REQUIRE(a.n_slices() == b.n_slices());

  for (size_t i = 0; i < a.n_elem(); ++i)
  {
    if (std::abs(a[i]) < tolerance / 2)
      REQUIRE(b[i] == Approx(0.0).margin(tolerance / 2));
    else
      REQUIRE(a[i] == Approx(b[i]).epsilon(tolerance / 100));
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
    FAIL("The matrices are equal.");
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
    FAIL("The matrices are equal.");
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
    FAIL("The matrices are equal.");
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
