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
    // set size of the output
    output.set_size(input.n_rows, input.n_cols);

    using PairType = std::pair<size_t, size_t>;
    // dimensions and indexes are saved as pairs inside this vector.
    std::vector<PairType> targets;
    // good elements are kept inside this vector.
    std::vector<double> elemsToKeep;

    if (columnMajor)
    {
      for (size_t row = 0; row < input.n_rows; ++row)
      {
        for (size_t col = 0; col < input.n_cols; ++col)
        {
          if (row == dimension)
          {
            if (input(row, col) == mappedValue ||
                std::isnan(input(row, col)))
            {
              targets.emplace_back(row, col);
            }
            else
            {
              elemsToKeep.push_back(input(row, col));
              output(row, col) = input(row, col);
            }
          }
          else
          {
            output(row, col) = input(row, col);
          }
        }
      }
    }
    else
    {
      for (size_t col = 0; col < input.n_cols; ++col)
      {
        for (size_t row = 0; row < input.n_rows; ++row)
        {
          if (col == dimension)
          {
            if (input(row, col) == mappedValue ||
                std::isnan(input(row, col)))
            {
              targets.emplace_back(row, col);
            }
            else
            {
              elemsToKeep.push_back(input(row, col));
              output(row, col) = input(row, col);
            }
          }
          else
          {
            output(row, col) = input(row, col);
          }
        }
      }
    }

    // calculate median
    const double median = arma::median(arma::vec(elemsToKeep));

    for (const PairType& target : targets)
    {
       output(target.first, target.second) = median;
    }
  }

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
