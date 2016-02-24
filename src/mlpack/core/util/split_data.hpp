#ifndef __MLPACK_CORE_UTIL_SPLIT_DATA_HPP
#define __MLPACK_CORE_UTIL_SPLIT_DATA_HPP

#include <mlpack/core.hpp>

#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

namespace mlpack {
namespace util {

/**
 *Split training data and test data, please define 
 *ARMA_USE_CXX11 to enable move of c++11
 *@param input input data want to split
 *@param label input label want to split
 *@param testRatio the ratio of test data
 *@param seed seed of the random device
 *@code
 *arma::mat trainData = loadData();
 *arma::Row<size_t> label = loadLabel();
 *std::random_device rd;
 *auto trainTest = TrainTestSplit(trainData, label, 0.25, rd());
 *@endcode
 */
template<typename T>
std::tuple<arma::Mat<T>, arma::Mat<T>,
arma::Row<size_t>, arma::Row<size_t>>
TrainTestSplit(arma::Mat<T> const &input,
               arma::Row<size_t> const &label,
               double testRatio,
               unsigned int seed = 0)
{
  size_t const testSize =
      static_cast<size_t>(input.n_cols * testRatio);
  size_t const trainSize = input.n_cols - testSize;

  arma::Mat<T> trainData(input.n_rows, trainSize);
  arma::Mat<T> testData(input.n_rows, testSize);
  arma::Row<size_t> trainLabel(trainSize);
  arma::Row<size_t> testLabel(testSize);

  std::vector<size_t> permutation(input.n_cols);
  std::iota(std::begin(permutation), std::end(permutation), 0);

  std::mt19937 gen(seed);
  std::shuffle(std::begin(permutation), std::end(permutation), gen);

  for(size_t i = 0; i != trainData.n_cols; ++i)
  {
     trainData.col(i) = input.col(permutation[i]);
     trainLabel(i) = label(permutation[i]);
  }

  for(size_t i = 0; i != testData.n_cols; ++i)
  {
     testData.col(i) = input.col(permutation[i + trainSize]);
     testLabel(i) = label(permutation[i + trainSize]);
  }

  return std::make_tuple(trainData, testData, trainLabel, testLabel);
}

} // namespace util
} // namespace mlpack

#endif
