/**
 * @file hmm_test_utils.hpp
 * @author Niteya Shah
 *
 * Helper Functions used in the execution of the CLI Range Search Test
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_MAIN_TESTS_RANGE_SEARCH_TEST_UTILS_HPP
#define MLPACK_TESTS_MAIN_TESTS_RANGE_SEARCH_TEST_UTILS_HPP

#include <boost/test/unit_test.hpp>
#include <mlpack/methods/range_search/rs_model.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
/*
* Convert a Model to String by calling the RSModel serialize function of the
* boost library and return the Model in String Form
* @param model - RSModel to be converted to string
*/
inline std::string ModelToString(RSModel* model)
{
  std::ostringstream oss;
  boost::archive::text_oarchive oa(oss);
  oa << model;
  return oss.str();
}

/*
* Check for 2 matrices of type vector<vector<double>> to ensure that their
* values dont differ by more than tolerance , default is 0.001%
* @param vec1 - vector 1 to be checked
* @param vec2 - vector 2 to be checked
* @param tolerance - difference in values in allowed
*/
inline void CheckMatrices(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2, float tolerance=1e-3)
{
  BOOST_REQUIRE_EQUAL(vec1.size() , vec2.size() );
  for (size_t i = 0; i < vec1.size(); i++)
  {
    BOOST_REQUIRE_EQUAL(vec1[i].size(), vec2[i].size() );
    std::sort(vec1[i].begin(), vec1[i].end());
    std::sort(vec2[i].begin(), vec2[i].end());
    for (size_t j = 0 ; j < vec1[i].size(); j++)
    {
      BOOST_REQUIRE_CLOSE(vec1[i][j], vec2[i][j], tolerance);
    }
  }
}

/*
* Check for 2 matrices of type vector<vector<size_t>> to ensure that their
* values match
* @param vec1 - vector 1 to be checked
* @param vec2 - vector 2 to be checked
*/
inline void CheckMatrices(std::vector<std::vector<size_t>> vec1, std::vector<std::vector<size_t>> vec2)
{

  BOOST_REQUIRE_EQUAL(vec1.size() , vec2.size() );
  for (size_t i = 0; i < vec1.size(); i++)
  {
    BOOST_REQUIRE_EQUAL(vec1[i].size(), vec2[i].size() );
    std::sort(vec1[i].begin(), vec1[i].end());
    std::sort(vec2[i].begin(), vec2[i].end());
    for (size_t j = 0; j < vec1[i].size(); j++)
    {
      BOOST_REQUIRE_EQUAL(vec1[i][j], vec2[i][j]);
    }
  }
}

/*
* Load A csv file into a vector of vector of templated datatype (code strips ',')
* by splitting on '\n' for lines and spaces for parts of a line
* @param path - path of the string
*/
template<typename T>
std::vector<std::vector<T>> ReadData(const std::string& path)
{
  std::ifstream ifs(path);
  std::vector<std::vector<T>> table;
  std::string line;
  while (std::getline(ifs, line))
  {
    std::vector<T> numbers;
    T n;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream stm(line) ;
    while ( stm >> n )
      numbers.push_back(n);
    table.push_back(numbers);
  }
  return table;
}

#endif
