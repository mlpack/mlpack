/**
 * @file core/data/imputation_methods/custom_imputation.hpp
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

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A simple custom imputation class, which replaces missing values with a
 * predefined value.
 */
template<typename ElemType = double>
class CustomImputation
{
 public:
  CustomImputation(const ElemType customValue):
      customValue(customValue)
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
  template<typename MatType>
  void Impute(MatType& input,
              const typename MatType::elem_type& missingValue,
              const size_t dimension,
              const bool columnMajor = true)
  {
    typedef typename MatType::elem_type T;

    // Here we can use .replace() directly.
    if (columnMajor)
      input.row(dimension).replace(missingValue, (T) customValue);
    else
      input.col(dimension).replace(missingValue, (T) customValue);
  }

 private:
  // A user-defined value that the user wants to replace missing values with.
  ElemType customValue;
}; // class CustomImputation

} // namespace mlpack

#endif
