/**
 * @file custom_imputation.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the empty CustomImputation class.
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
   * The result is saved to the output. Custom value must be set when
   * initializing the CustomImputation object.
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

    // replace the target value to custom value
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
              output(row, col) = customValue;
            }
            else
            {
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
      for (size_t col = 0; col < input.n_cols; ++ col)
      {
        for (size_t row = 0; row < input.n_rows; ++row)
        {
          if (col == dimension)
          {
            if (input(row, col) == mappedValue ||
                std::isnan(input(row, col)))
            {
              output(row, col) = customValue;
            }
            else
            {
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
