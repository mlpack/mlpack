/**
 * @file binarize.hpp
 * @author Keon Kim
 *
 * Defines Binarize(), a utility function, sets values to 0 or 1
 * to a given threshold.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_BINARIZE_HPP
#define MLPACK_CORE_DATA_BINARIZE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

/**
 * Given an input dataset and threshold, set values greater than threshold to
 * 1 and values less than or equal to the threshold to 0. This overload applies
 * the changes to all dimensions.
 *
 * @code
 * arma::Mat<double> input = loadData();
 * arma::Mat<double> output;
 * double threshold = 0.5;
 *
 * // Binarize the whole Matrix. All positive values in will be set to 1 and
 * // the values less than or equal to 0.5 will become 0.
 * Binarize<double>(input, output, threshold);
 * @endcode
 *
 * @param input Input matrix to Binarize.
 * @param output Matrix you want to save binarized data into.
 * @param threshold Threshold can by any number.
 */
template<typename T>
void Binarize(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const double threshold)
{
  output.copy_size(input);

  const int totalElems = static_cast<int>(input.n_elem);
  const T *inPtr = input.memptr();
  T *outPtr = output.memptr();

  #pragma omp parallel for
  for (int i = 0; i < totalElems; ++i)
  {
    if (inPtr[i] > threshold)
      outPtr[i] = 1;
    else
      outPtr[i] = 0;
  }
}

/**
 * Given an input dataset and threshold, set values greater than threshold to
 * 1 and values less than or equal to the threshold to 0. This overload takes
 * a dimension and applys the changes to the given dimension.
 *
 * @code
 * arma::Mat<double> input = loadData();
 * arma::Mat<double> output;
 * double threshold = 0.5;
 * size_t dimension = 0;
 *
 * // Binarize the first dimension. All positive values in the first dimension
 * // will be set to 1 and the values less than or equal to 0 will become 0.
 * Binarize<double>(input, output, threshold, dimension);
 * @endcode
 *
 * @param input Input matrix to Binarize.
 * @param output Matrix you want to save binarized data into.
 * @param threshold Threshold can by any number.
 * @param dimension Feature to apply the Binarize function.
 */
template<typename T>
void Binarize(const arma::Mat<T>& input,
              arma::Mat<T>& output,
              const double threshold,
              const size_t dimension)
{
  output = input;
  const int totalCols = static_cast<int>(input.n_cols);

  #pragma omp parallel for
  for (int i = 0; i < totalCols; ++i)
  {
    if (input(dimension, i) > threshold)
      output(dimension, i) = 1;
    else
      output(dimension, i) = 0;
  }
}

} // namespace data
} // namespace mlpack

#endif
