/**
 * @file median_strategy.hpp
 * @author Keon Kim
 *
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_STRATEGY_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_STRATEGY_HPP


#include <mlpack/core.hpp>


using namespace std;

namespace mlpack {
namespace data {

class MedianStrategy
{
 public:

  template <typename MatType>
  void Impute(const MatType &input,
              MatType &output,
              const size_t dimension,
              const size_t index,
              const bool transpose = true)
  {
    if (transpose)
    {
      MatType medianMat = arma::median(input, 1);
      output(dimension, index) = medianMat(dimension);
    }
    else
    {
      MatType medianMat = arma::median(input, 0);
      output(index, dimension) = medianMat(index);
    }
  }
};

} // namespace data
} // namespace mlpack

#endif
