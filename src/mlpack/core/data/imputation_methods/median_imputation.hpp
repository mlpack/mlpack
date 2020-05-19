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
   * replaces it with the median of the given dimension. The result is
   * overwritten to the input matrix.
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
    using PairType = std::pair<size_t, size_t>;
    // dimensions and indexes are saved as pairs inside this vector.
    std::vector<PairType> targets;
    // good elements are kept inside this vector.
    std::vector<double> elemsToKeep;

    if (columnMajor)
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(dimension, i) == mappedValue ||
            std::isnan(input(dimension, i)))
        {
          targets.emplace_back(dimension, i);
        }
        else
        {
          elemsToKeep.push_back(input(dimension, i));
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
          targets.emplace_back(i, dimension);
        }
        else
        {
           elemsToKeep.push_back(input(i, dimension));
        }
      }
    }

    // calculate median
    const double median = arma::median(arma::vec(elemsToKeep));

    for (const PairType& target : targets)
    {
       input(target.first, target.second) = median;
    }
  }
}; // class MedianImputation

} // namespace data
} // namespace mlpack

#endif
