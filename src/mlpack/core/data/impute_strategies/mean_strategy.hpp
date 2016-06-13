/**
 * @file mean_strategy.hpp
 * @author Keon Kim
 *
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_STRATEGY_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_STRATEGY_HPP

#include <mlpack/core.hpp>


using namespace std;

namespace mlpack {
namespace data {

class MeanStrategy
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
      MatType meanMat = arma::mean(input, 1);
      output(dimension, index) = meanMat(dimension);
    }
    else
    {
      MatType meanMat = arma::mean(input, 0);
      output(index, dimension) = meanMat(index);
    }
 }
};

} // namespace data
} // namespace mlpack

#endif
