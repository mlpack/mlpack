/**
 * @file median_imputation.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the MedianImputation class.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP

#include <mlpack/core.hpp>

using namespace std;

namespace mlpack {
namespace data {

/**
 * A simple median imputation
 * replace missing value with middle or average of middle values
 */
template <typename T>
class MedianImputation
{
 public:
  void Apply (const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const T& mappedValue,
              const size_t dimension,
              const bool transpose = true)
  {
    //initiate output
    output = input;

    if (transpose)
    {
      arma::Mat<T> medianMat = arma::median(input, 1);
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(dimension, i) == mappedValue)
        {
          output(dimension, i) = medianMat(0, i);
        }
      }
    }
    else
    {
      arma::Mat<T> medianMat = arma::median(input, 0);
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (input(i, dimension) == mappedValue)
        {
          output(i, dimension) = medianMat(i, 0);
        }
      }
    }
  }
}; // class MeanImputation

} // namespace data
} // namespace mlpack

#endif
