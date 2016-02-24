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

namespace details{

template<typename T>
inline
arma::Mat<T> createData(arma::Mat<T> const &input,
                        size_t dataSize)
{
  return arma::Mat<T>(input.n_rows, dataSize);
}

template<typename T>
inline
arma::Cube<T> createData(arma::Cube<T> const &input,
                         size_t dataSize)
{
  return arma::Cube<T>(input.n_rows, input.n_cols, dataSize);
}

template<typename T>
inline
void extractData(arma::Mat<T> const &input, arma::Mat<T> &output,
                 size_t inputIndex, size_t outputIndex)
{
  output.col(outputIndex) = input.col(inputIndex);
}

template<typename T>
inline
void extractData(arma::Cube<T> const &input, arma::Cube<T> &output,
                 size_t inputIndex, size_t outputIndex)
{
  output.slice(outputIndex) = input.slice(inputIndex);
}

template<typename T>
inline
size_t extractSize(arma::Mat<T> const &input)
{
  return input.n_cols;
}

template<typename T>
inline
size_t extractSize(arma::Cube<T> const &input)
{
  return input.n_slices;
}

}

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
std::tuple<T, T,
arma::Row<size_t>, arma::Row<size_t>>
TrainTestSplit(T const &input,
               arma::Row<size_t> const &label,
               double testRatio,
               unsigned int seed = 0)
{
  size_t const testSize =
      static_cast<size_t>(details::extractSize(input) * testRatio);
  size_t const trainSize = details::extractSize(input) - testSize;

  T trainData = details::createData(input, trainSize);
  T testData = details::createData(input, testSize);
  arma::Row<size_t> trainLabel(trainSize);
  arma::Row<size_t> testLabel(testSize);

  std::vector<size_t> permutation(details::extractSize(input));
  std::iota(std::begin(permutation), std::end(permutation), 0);

  std::mt19937 gen(seed);
  std::shuffle(std::begin(permutation), std::end(permutation), gen);

  for(size_t i = 0; i != trainSize; ++i)
  {
    details::extractData(input, trainData, permutation[i], i);
    trainLabel(i) = label(permutation[i]);
  }

  for(size_t i = 0; i != testSize; ++i)
  {
    details::extractData(input, testData,
                         permutation[i + trainSize], i);
    testLabel(i) = label(permutation[i + trainSize]);
  }

  return std::make_tuple(trainData, testData, trainLabel, testLabel);
}

} // namespace util
} // namespace mlpack

#endif
