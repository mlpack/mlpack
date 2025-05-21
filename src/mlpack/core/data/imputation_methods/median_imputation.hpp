/**
 * @file core/data/imputation_methods/median_imputation.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the MedianImputation class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEDIAN_IMPUTATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This is a class implementation of simple median imputation: replace missing
 * value with the median of non-missing values.
 */
class MedianImputation
{
 public:
  /**
   * Impute function searches through the input looking for mappedValue and
   * replaces it with the median of the given dimension. The result is
   * overwritten to the input matrix.
   *
   * @param input Matrix that contains mappedValue.
   * @param mappedValue Value that the user wants to get rid of.
   * @param dimension Index of the dimension of the mappedValue.
   * @param columnMajor State of whether the input matrix is columnMajor or not.
   */
  template<typename MatType>
  static void Impute(MatType& input,
                     const typename MatType::elem_type& missingValue,
                     const size_t dimension,
                     const bool columnMajor = true)
  {
    static_assert(!IsSparse<MatType>::value, "MedianImputation::Impute(): "
        "sparse matrix imputation is not supported; use a dense matrix "
        "instead!");

    typedef typename MatType::elem_type ElemType;

    // If mappedValue is NaN, Armadillo does not quite provide the tools we need
    // so we have to do our own implementation.  Otherwise, we can directly use
    // Armadillo pretty easily.
    ElemType medianValue;
    MatType tmp;
    if (std::isnan(missingValue))
    {
      if (columnMajor)
        tmp = input.row(dimension);
      else
        tmp = input.col(dimension).t();

      tmp.shed_cols(find_nan(tmp));
    }
    else
    {
      typedef typename GetUColType<MatType>::type UCol;
      if (columnMajor)
      {
        tmp = input.submat(UCol({ dimension }),
            find(input.row(dimension) != missingValue));
      }
      else
      {
        tmp = input.submat(
            find(input.col(dimension) != missingValue), UCol({ dimension }));
      }
    }

    // Compute the median on the extracted elements.
    if (tmp.is_empty())
    {
      throw std::invalid_argument("MedianImputation::Impute(): no non-missing "
          "elements; cannot compute median!");
    }
    medianValue = median(vectorise(tmp));

    if (columnMajor)
      input.row(dimension).replace(missingValue, medianValue);
    else
      input.col(dimension).replace(missingValue, medianValue);
  }
}; // class MedianImputation

} // namespace mlpack

#endif
