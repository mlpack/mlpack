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
//Templated Check for 2 matrices of type nested vectors
template<typename N>
inline void CheckMatrices( std::vector<std::vector<N>> vec1, std::vector<std::vector<N>> vec2, float tolerance=1e-3)
{

  BOOST_REQUIRE_EQUAL( vec1.size() , vec2.size() );
  for ( size_t i = 0; i < vec1.size() ; i++)
  {
    BOOST_REQUIRE_EQUAL( vec1[i].size(), vec2[i].size() );
    std::sort(vec1[i].begin(), vec1[i].end());
    std::sort(vec2[i].begin(), vec2[i].end());
    for (size_t j = 0 ; j < vec1[i].size() ; j++)
    {
      BOOST_REQUIRE_CLOSE(static_cast<float>(vec1[i][j]), static_cast<float>(vec2[i][j]), tolerance);
    }
  }
}
//Load A csv file into a vector of vector of templated datatype (code strips ',')
template<typename T>
std::vector<std::vector<T>> ReadData(std::string const& path)
{
    std::ifstream ifs(path);
    std::vector<std::vector<T>> table;
    std::string line;
    while (std::getline(ifs, line))
    {
      std::vector<T> numbers ;
      T n ;
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream stm(line) ;
      while( stm >> n ) numbers.push_back(n) ;
      table.push_back(numbers);
    }
    return table;
}
#endif
