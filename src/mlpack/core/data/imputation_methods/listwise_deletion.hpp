/**
 * @file listwise_deletion.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the empty ListwiseDeletion class.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_LISTWISE_DELETION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_LISTWISE_DELETION_HPP

#include <mlpack/core.hpp>

using namespace std;

namespace mlpack {
namespace data {

/**
 * complete-case analysis.
 * Removes all data for a case that has one or more missing values.
 */
template <typename T>
class ListwiseDeletion
{
 public:
  void Apply(const arma::Mat<T>& input,
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
         if (input(dimension, i) == mappedValue)
         {
           output.shed_col(i - count);
           count++;
         }
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_rows; ++i)\
      {
        if (input(i, dimension) == mappedValue)
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
