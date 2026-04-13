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
  template<typename T>
  static void Impute(arma::Mat<T>& input,
                     const T& missingValue,
                     const size_t dimension,
                     const bool columnMajor = true)
  {
    T medianValue;

    #if ARMA_VERSION_MAJOR < 14 || \
        (ARMA_VERSION_MAJOR == 14 && ARMA_VERSION_MINOR < 6)
      // This is used when omit_nan() is not available.

      // If mappedValue is NaN, Armadillo does not quite provide the tools we
      // need so we have to do our own implementation.  Otherwise, we can
      // directly use Armadillo pretty easily.
      arma::Mat<T> tmp;
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
        if (columnMajor)
        {
          tmp = input.submat(arma::uvec({ dimension }),
              find(input.row(dimension) != missingValue));
        }
        else
        {
          tmp = input.submat(
              find(input.col(dimension) != missingValue),
              arma::uvec({ dimension }));
        }
      }

      // Compute the median on the extracted elements.
      if (tmp.is_empty())
      {
        throw std::invalid_argument("MedianImputation::Impute(): no non-missing "
            "elements; cannot compute median!");
      }
      medianValue = median(vectorise(tmp));
    #else
      if (std::isnan(missingValue))
      {
        if (columnMajor)
          medianValue = median(omit_nan(input.row(dimension)));
        else
          medianValue = median(omit_nan(input.col(dimension)));
      }
      else
      {
        if (columnMajor)
        {
          medianValue = median(vectorise(input.submat(arma::uvec({ dimension }),
              find(input.row(dimension) != missingValue))));
        }
        else
        {
          medianValue = median(vectorise(input.submat(
              find(input.col(dimension) != missingValue),
              arma::uvec({ dimension }))));
        }
      }
    #endif

    if (columnMajor)
      input.row(dimension).replace(missingValue, medianValue);
    else
      input.col(dimension).replace(missingValue, medianValue);
  }

  /**
   * Impute function searches through the input looking for mappedValue and
   * replaces it with the median of the given dimension. The result is
   * overwritten to the input matrix.
   *
   * This overload is used for Bandicoot, where omit_nan() is not available
   * (yet).
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
    typedef typename GetUColType<MatType>::type UCol;
    ElemType medianValue;
    if (columnMajor)
    {
      UCol indices;
      if (std::isnan(missingValue))
        indices = find_nonnan(input.row(dimension));
      else
        indices = find(input.row(dimension) != missingValue);

      medianValue = median(vectorise(input.submat(UCol({ dimension }),
          indices)));
    }
    else
    {
      UCol indices;
      if (std::isnan(missingValue))
        indices = find_nonnan(input.col(dimension));
      else
        indices = find(input.col(dimension) != missingValue);

      medianValue = median(vectorise(input.submat(indices,
          UCol({ dimension }))));
    }

    if (columnMajor)
      input.row(dimension).replace(missingValue, medianValue);
    else
      input.col(dimension).replace(missingValue, medianValue);
  }
}; // class MedianImputation

} // namespace mlpack

#endif
