/**
 * @file median_imputation.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the MedianImputation class.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {
/**
 * This is a class implementation of simple median imputation.
 * replace missing value with middle or average of middle values
 * @tparam T Type of armadillo matrix
 */
template <typename T>
class MedianImputation
{
 public:
  /**
   * Impute function searches through the input looking for mappedValue and
   * replaces it with the median of the given dimension. The result is saved
   * to the output.
   *
   * @param input Matrix that contains mappedValue.
   * @param output Matrix that the result will be saved into.
   * @param mappedValue Value that the user wants to get rid of.
   * @param dimension Index of the dimension of the mappedValue.
   * @param columnMajor State of whether the input matrix is columnMajor or not.
   */
  void Impute(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const T& mappedValue,
              const size_t dimension,
              const bool columnMajor = true)
  {
    //initiate output
    output = input;

    if (columnMajor)
    {
      arma::Mat<T> medianMat = arma::median(input, 1);
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(dimension, i) == mappedValue ||
            std::isnan(input(dimension, i)))
        {
          output(dimension, i) = medianMat(dimension, 0);
        }
      }
    }
    else
    {
      arma::Mat<T> medianMat = arma::median(input, 0);
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (input(i, dimension) == mappedValue ||
            std::isnan(input(i, dimension)))
        {
          output(i, dimension) = medianMat(0, dimension);
        }
      }
    }
  }
}; // class MedianImputation

} // namespace data
} // namespace mlpack

#endif
