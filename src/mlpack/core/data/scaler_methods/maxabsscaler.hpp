/**
 * @file maxabsscaler.hpp
 * @author Jeffin Sam
 *
 * MaxAbsScaler class to scale features.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MAX_ABS_SCALE_HPP
#define MLPACK_CORE_DATA_MAX_ABS_SCALE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {
/**
 * A simple MaxAbs Scaler class.
 *
 * Given an input dataset this class helps you to scale each
 * feature by its maximum absolute value.
 *
 * @code
 * arma::Mat<double> input = loadData();
 * arma::Mat<double> output;
 *
 * // Scale the features using
 * MaxAbsScaler<double> scale;
 * scale.Tranform(input, output);
 *
 * // The input can be retransformed using
 * scale.InverseTransform(output, input);
 * @endcode
 *
 */
template <typename T>
class MaxAbsScaler
{
 public:
  /**
  * Function to scale Features.
  *
  * @param input Dataset to scale features.
  * @param output Output matrix with scaled features.
  */
  void Transform(arma::Mat<T>& input,arma::Mat<T>& output)
  {
    output.set_size(size(input));
    itemMin = arma::min(input, 1);
    itemMax = arma::max(input, 1);
    scale = arma::max(arma::abs(itemMin), arma::abs(itemMax));
    // Handline Zeroes in scale vector
    for (size_t i = 0; i < scale.n_elem; i++)
    {
      if (scale(i) == 0)
      {
        scale(i) = 1;
      }
    }
    for (size_t i = 0; i < input.n_rows; i++)
    {
      for (size_t j = 0; j < input.n_cols; j++)
      {
        output(i, j) = input(i, j) / scale(i);
      }
    }
  }

  /**
  * Function to retrive original dataset.
  *
  * @param input Scaled dataset.
  * @param output Output matrix with original Dataset.
  */
  void InverseTransform(arma::Mat<T>& input, arma::Mat<T>& output)
  {
    output.set_size(size(input));
    for (size_t i = 0; i < input.n_rows; i++)
    {
      for (size_t j = 0; j < input.n_cols; j++)
      {
        output(i, j) = input(i, j) * scale(i);
      }
    }
  }

  //! Get the Min row vector.
  const arma::colvec& ItemMin() const { return itemMin; }
  //! Get the Max row vector.
  const arma::colvec& ItemMax() const { return itemMax; }
  //! Get the Scale row vector.
  const arma::colvec& Scale() const { return scale; }

 private:
  // Vector which holds minimum of each feature.
  arma::colvec itemMin;
  // Vector which holds maximum of each feature.
  arma::colvec itemMax;
  // Vector which is used to scale up each feature.
  arma::colvec scale;
}; // class MaxAbsScaler

} // namespace data
} // namespace mlpack

#endif
