/**
 * @file listwise_deletion.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the empty ListwiseDeletion class.
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
   * remove the whole row or column. The result is saved to the output.
   *
   * @param input Matrix that contains mappedValue.
   * @param output Matrix that the result will be saved into.
   * @param mappedValue Value that the user wants to get rid of.
   * @param dimension Index of the dimension of the mappedValue.
   * @param transpose State of whether the input matrix is transposed or not.
   */
  void Impute(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const T& mappedValue,
              const size_t dimension,
              const bool transpose = true)
  {
    // initiate output
    output = input;
    size_t count = 0;

    if (transpose)
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
         if (input(dimension, i) == mappedValue ||
             std::isnan(input(dimension, i)))
         {
           output.shed_col(i - count);
           count++;
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
           output.shed_row(i - count);
           count++;
        }
      }
    }
  }
}; // class ListwiseDeletion

} // namespace data
} // namespace mlpack

#endif
