/**
 * @file mean_strategy.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the MeanStrategy class.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_STRATEGY_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_STRATEGY_HPP

#include <mlpack/core.hpp>


using namespace std;

namespace mlpack {
namespace data {

/**
 * The MeanStrategy
 */
class MeanStrategy
{
 public:

  /**
   * Computes mean, excluding NaN or target missing variables
   *
   * TODO: write docs for parameters
   * @param input
   * @param output
   * @param dimension
   * @param index
   * @param transpose
   */
  template <typename MatType>
  void Impute(const MatType &input,
              MatType &output,
              const size_t dimension,
              const size_t index,
              const bool transpose = true)
  {
    if (transpose)
    {
      // TODO: The mean must be calculated
      // without NaN or target missing variable.
      MatType meanMat = arma::mean(input, 1);
      output(dimension, index) = meanMat(dimension);
    }
    else
    {
      // TODO: The mean must be calculated
      // without NaN or target missing variable.
      MatType meanMat = arma::mean(input, 0);
      output(index, dimension) = meanMat(index);
    }
 }
}; // class MeanStrategy

} // namespace data
} // namespace mlpack

#endif
