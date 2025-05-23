/**
 * @file core/data/imputation_methods/listwise_deletion.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the empty ListwiseDeletion class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_LISTWISE_DELETION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_LISTWISE_DELETION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A complete-case analysis to remove the columns containing missingValue.
 * Removes all data for a case that has one or more missing values.
 */
class ListwiseDeletion
{
 public:
  /**
   * Impute function searches through the input looking for mappedValue and
   * remove the whole row or column. The result is overwritten to the input.
   *
   * @param input Matrix that contains mappedValue.
   * @param missingValue Value that the user wants to get rid of.
   * @param dimension Index of the dimension to search for missingValue.
   * @param columnMajor State of whether the input matrix is columnMajor or not.
   */
  template<typename MatType>
  void Impute(MatType& input,
              const typename MatType::elem_type& missingValue,
              const size_t dimension,
              const bool columnMajor = true)
  {
    static_assert(!IsSparse<MatType>::value, "ListwiseDeletion::Impute(): "
        "sparse matrix imputation is not supported; use a dense matrix "
        "instead!");

    if (std::isnan(missingValue))
    {
      if (columnMajor)
        input.shed_cols(find_nan(input.row(dimension)));
      else
        input.shed_rows(find_nan(input.col(dimension)));
    }
    else
    {
      if (columnMajor)
        input.shed_cols(find(input.row(dimension) == missingValue));
      else
        input.shed_rows(find(input.col(dimension) == missingValue));
    }
  }
}; // class ListwiseDeletion

} // namespace mlpack

#endif
