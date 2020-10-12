/**
 * @file tests/main_tests/range_search_utils.hpp
 * @author Niteya Shah
 *
 * Helper functions used in the execution of the Range Search test.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_MAIN_TESTS_RANGE_SEARCH_TEST_UTILS_HPP
#define MLPACK_TESTS_MAIN_TESTS_RANGE_SEARCH_TEST_UTILS_HPP

#include <mlpack/methods/range_search/rs_model.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "../catch.hpp"

/**
 * Convert a model to a string using the text_oarchive of boost::serialization.
 *
 * @param model RSModel to be converted to string.
 */
inline std::string ModelToString(RSModel* model)
{
  std::ostringstream oss;
  boost::archive::text_oarchive oa(oss);
  oa << model;
  return oss.str();
}

/**
 * Check that 2 matrices of type vector<vector<double>> are close to equal,
 * using the given tolerance.
 *
 * @param vec1 First vector to compare.
 * @param vec2 Second vector to compare.
 * @param tolerance Allowed tolerance for values.
 */
inline void CheckMatrices(std::vector<std::vector<double>>& vec1,
                          std::vector<std::vector<double>>& vec2,
                          const double tolerance = 1e-3)
{
  REQUIRE(vec1.size()  == vec2.size());
  for (size_t i = 0; i < vec1.size(); ++i)
  {
    REQUIRE(vec1[i].size() == vec2[i].size());
    std::sort(vec1[i].begin(), vec1[i].end());
    std::sort(vec2[i].begin(), vec2[i].end());
    for (size_t j = 0 ; j < vec1[i].size(); ++j)
    {
      REQUIRE(vec1[i][j] == Approx(vec2[i][j]).epsilon(tolerance));
    }
  }
}

/**
 * Check that 2 matrices of type vector<vector<size_t>> are equal.
 *
 * @param vec1 First vector to compare.
 * @param vec2 Second vector to compare.
 */
inline void CheckMatrices(std::vector<std::vector<size_t>>& vec1,
                          std::vector<std::vector<size_t>>& vec2)
{
  REQUIRE(vec1.size()  == vec2.size());
  for (size_t i = 0; i < vec1.size(); ++i)
  {
    REQUIRE(vec1[i].size() == vec2[i].size());
    std::sort(vec1[i].begin(), vec1[i].end());
    std::sort(vec2[i].begin(), vec2[i].end());
    for (size_t j = 0; j < vec1[i].size(); ++j)
    {
      REQUIRE(vec1[i][j] == vec2[i][j]);
    }
  }
}

/**
 * Load a CSV file into a vector of vector with a templated datatype.  Any ','
 * characters are stripped from the input; lines are split on '\n' and elements
 * of each line are split on spaces.
 *
 * @param filename Name of the file to load.
 */
template<typename T>
std::vector<std::vector<T>> ReadData(const std::string& filename)
{
  std::ifstream ifs(filename);
  std::vector<std::vector<T>> table;
  std::string line;
  while (std::getline(ifs, line))
  {
    std::vector<T> numbers;
    T n;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream stm(line);
    while (stm >> n)
      numbers.push_back(n);
    table.push_back(numbers);
  }

  return table;
}

#endif
