/**
 * @file custom_imputation.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the empty CustomImputation class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_CUSTOM_IMPUTATION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_CUSTOM_IMPUTATION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {
/**
 * A simple custom imputation class
 * @tparam T Type of armadillo matrix
 */
template <typename T>
class CustomImputation
{
 public:
  CustomImputation(T customValue):
      customValue(std::move(customValue))
  {
    // nothing to initialize here
  }

  /**
   * Impute function searches through the input looking for mappedValue and
   * replaces it with the user-defined custom value of the given dimension.
   * The result is overwritten to the input, not creating any copy. Custom value
   * must be set when initializing the CustomImputation object.
   *
   * @param input Matrix that contains mappedValue.
   * @param mappedValue Value that the user wants to get rid of.
   * @param dimension Index of the dimension of the mappedValue.
   * @param columnMajor State of whether the input matrix is columnMajor or not.
   */
  void Impute(arma::Mat<T>& input,
              const T& mappedValue,
              const size_t dimension,
              const bool columnMajor = true)
  {
    // replace the target value to custom value
    if (columnMajor)
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(dimension, i) == mappedValue ||
            std::isnan(input(dimension, i)))
        {
          input(dimension, i) = customValue;
        }
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (input(i, dimension) == mappedValue ||
            std::isnan(input(i, dimension)))
        {
          input(i, dimension) = customValue;
        }
      }
    }
  }

 private:
  //! A user-defined value that the user wants to replace missing values with.
  T customValue;
}; // class CustomImputation

} // namespace data
} // namespace mlpack

#endif
