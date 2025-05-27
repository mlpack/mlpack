/**
 * @file core/data/imputation_methods/mean_imputation.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the MeanImputation class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_IMPUTATION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MEAN_IMPUTATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A simple mean imputation class.
 */
class MeanImputation
{
 public:
  /**
   * Impute function searches through the input looking for mappedValue and
   * replaces it with the mean of the given dimension. The result is overwritten
   * to the input matrix.
   *
   * @param input Matrix that contains missingValue.
   * @param missingValue Value to replace with the mean.
   * @param dimension Index of the dimension to impute.
   * @param columnMajor If true, `input` is column major.
   */
  template<typename T>
  static void Impute(arma::Mat<T>& input,
                     const T& missingValue,
                     const size_t dimension,
                     const bool columnMajor = true)
  {
    T sum = 0;
    size_t elems = 0; // Excluding missingValue.

    // Different implementations if we are searching for NaN.
    if (std::isnan(missingValue))
    {
      if (columnMajor)
      {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < input.n_cols; ++i)
        {
          if (!std::isnan(input(dimension, i)))
          {
            ++elems;
            sum += input(dimension, i);
          }
        }
      }
      else
      {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < input.n_rows; ++i)
        {
          if (!std::isnan(input(i, dimension)))
          {
            ++elems;
            sum += input(i, dimension);
          }
        }
      }
    }
    else
    {
      if (columnMajor)
      {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < input.n_cols; ++i)
        {
          if (input(dimension, i) != missingValue)
          {
            ++elems;
            sum += input(dimension, i);
          }
        }
      }
      else
      {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < input.n_rows; ++i)
        {
          if (input(i, dimension) != missingValue)
          {
            ++elems;
            sum += input(i, dimension);
          }
        }
      }
    }

    if (elems == 0)
    {
      throw std::invalid_argument("MeanImputation::Impute(): no non-missing "
          "elements; cannot compute mean!");
    }

    // Now compute the mean.
    const double mean = sum / elems;

    // Replace all values with the computed mean.
    if (columnMajor)
      input.row(dimension).replace(missingValue, mean);
    else
      input.col(dimension).replace(missingValue, mean);
  }

  template<typename MatType>
  static void Impute(MatType& input,
                     const typename MatType::elem_type& missingValue,
                     const size_t dimension,
                     const bool columnMajor = true)
  {
    static_assert(!IsSparse<MatType>::value, "MeanImputation::Impute(): sparse "
        "matrix imputation is not supported; use a dense matrix instead!");

    typedef typename MatType::elem_type ElemType;

    // This less efficient implementation uses Armadillo (or Bandicoot)
    // directly, but at the cost of more memory.
    ElemType meanValue;
    MatType tmp;
    if (std::isnan(missingValue))
    {
      if (columnMajor)
        tmp = input.row(dimension);
      else
        tmp = input.col(dimension).t(); // make sure it is a row vector

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

    // Compute the mean on the subset of valid elements.
    if (tmp.is_empty())
    {
      throw std::invalid_argument("MeanImputation::Impute(): no non-missing "
          "elements; cannot compute mean!");
    }
    meanValue = mean(vectorise(tmp));

    // Now impute the computed mean value.
    if (columnMajor)
      input.row(dimension).replace(missingValue, meanValue);
    else
      input.col(dimension).replace(missingValue, meanValue);
  }
}; // class MeanImputation

} // namespace mlpack

#endif
