/**
 * @file listwise_deletion.hpp
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

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {
/**
 * A complete-case analysis to remove the values containing mappedValue.
 * Removes all data for a case that has one or more missing values.
 * @tparam T Type of armadillo matrix
 */
template <typename T>
class ListwiseDeletion
{
 public:
  /**
   * Impute function searches through the input looking for mappedValue and
   * remove the whole row or column. The result is overwritten to the input.
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
    std::vector<arma::uword> colsToKeep;

    if (columnMajor)
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
         if (!(input(dimension, i) == mappedValue ||
             std::isnan(input(dimension, i))))
         {
           colsToKeep.push_back(i);
         }
      }
      input = input.cols(arma::uvec(colsToKeep));
    }
    else
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (!(input(i, dimension) == mappedValue ||
             std::isnan(input(i, dimension))))
        {
           colsToKeep.push_back(i);
        }
      }
      input = input.rows(arma::uvec(colsToKeep));
    }
  }
}; // class ListwiseDeletion

} // namespace data
} // namespace mlpack

#endif
