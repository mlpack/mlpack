/**
 * @file Standardscaler.hpp
 * @author Jeffin Sam
 *
 * StandardScaler class to scale features.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STANDARD_SCALE_HPP
#define MLPACK_CORE_DATA_STANDARD_SCALE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {
/**
 * A simple Standard Scaler class
 *
 * Given an input dataset this class helps you to Standardize features
 * by removing the mean and scaling to unit variance.
 *
 * @code
 * arma::Mat<double> input = loadData();
 * arma::Mat<double> output;
 *
 * // Scale the features.
 * StandardScaler<double> scale;
 * scale.Tranform(input, output);
 *
 * // Retransform the input.
 * scale.InverseTransform(output, input);
 * @endcode
 */
class StandardScaler
{
 public:

  /**
  * Function to scale Features.
  *
  * @param input Dataset to scale features.
  * @param output Output matrix with scaled features.
  */
  template<typename MatType>
  void Transform(const MatType& input, MatType& output)
  {
    output.copy_size(input);
    itemMean = arma::mean(input, 1);
    itemStdev = arma::stddev(input, 1, 1);

    // Handline Zeroes in scale vector.
    for (size_t i = 0; i < itemStdev.n_elem; i++)
    {
      if (itemStdev(i) == 0)
      {
        itemStdev(i) = 1;
      }
    }
    for (size_t i = 0; i < input.n_rows; i++)
    {
      for (size_t j = 0; j < input.n_cols; j++)
      {
        output(i, j) = input(i, j) - itemMean(i);
        output(i, j) /= itemStdev(i);
      }
    }
  }

  /**
  * Function to retrive original dataset.
  *
  * @param input Scaled dataset.
  * @param output Output matrix with original Dataset.
  */
  template<typename MatType>
  void InverseTransform(const MatType& input, MatType& output)
  {
    output.copy_size(input);
    for (size_t i = 0; i < input.n_rows; i++)
    {
      for (size_t j = 0; j < input.n_cols; j++)
      {
        output(i, j) = input(i, j) * itemStdev(i);
        output(i, j) += itemMean(i);
      }
    }
  }

  //! Get the Mean row vector.
  const arma::colvec& ItemMean() const { return itemMean; }
  //! Get the Standard Devation row vector.
  const arma::colvec& ItemStdev() const { return itemStdev; }

 private:
  // Vector which holds mean of each feature
  arma::colvec itemMean;
  // Vector which holds standard devation of each feature
  arma::colvec itemStdev;

}; // class StandardScaler

} // namespace data
} // namespace mlpack

#endif
