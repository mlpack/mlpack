/**
 * @file binarize.hpp
 * @author Keon Kim
 *
 * Defines Binarize(), a utility function, sets values to 0 or 1
 * to a given threshold.
 */
#ifndef MLPACK_CORE_DATA_BINARIZE_HPP
#define MLPACK_CORE_DATA_BINARIZE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {
/**
 * Given an input dataset and threshold, set values greater than threshold to
 * 1 and values less than or equal to the threshold to 0. This overload takes
 * a dimension and applys the changes to the given dimension.
 *
 * @code
 * arma::mat input = loadData();
 * double threshold = 0;
 * size_t dimension = 0;
 *
 * // Binarize the first dimension. All positive values in the first dimension
 * // will be set to 1 and the values less than or equal to 0 will become 0.
 * Binarize(input, threshold, dimension);
 * @endcode
 *
 * @param input Input matrix to Binarize.
 * @param threshold Threshold can by any number.
 * @param dimension Feature to apply the Binarize function.
 */
template<typename T>
void Binarize(arma::Mat<T>& input,
              const double threshold,
              const size_t dimension)
{
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    if (input(dimension, i) > threshold)
      input(dimension, i) = 1;
    else
      input(dimension, i) = 0;
  }
}

/**
 * Given an input dataset and threshold, set values greater than threshold to
 * 1 and values less than or equal to the threshold to 0. This overload applies
 * the changes to all dimensions.
 *
 * @code
 * arma::mat input = loadData();
 * double threshold = 0;
 *
 * // Binarize the whole Matrix. All positive values in will be set to 1 and
 * // the values less than or equal to 0 will become 0.
 * Binarize(input, threshold);
 * @endcode
 *
 * @param input Input matrix to Binarize.
 * @param threshold Threshold can by any number.
 */
template<typename T>
void Binarize(arma::Mat<T>& input,
              const double threshold)
{
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    for (size_t j = 0; j < input.n_rows; ++j)
    {
      if (input(i, j) > threshold)
        input(i, j) = 1;
      else
        input(i, j) = 0;
    }
  }
 }

template<typename T>
void Binarize(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const double threshold)
{
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    output.row(i) =
        arma::conv_to<arma::Mat<T>>::from(input.row(i) > threshold);
  }
}

template<typename T>
void Binarize(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const double threshold,
              const size_t dimension)
{
   output.row(dimension) =
      arma::conv_to<arma::Mat<T>>::from(input.row(dimension) > threshold);
}

} // namespace data
} // namespace mlpack

#endif
