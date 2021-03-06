/**
 * @file core/data/scaler_methods/standard_scaler.hpp
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
 * \[z = (x - u) / s\]
 *
 * where u is the mean of the training samples and s is the standard deviation
 * of the training samples.
 *
 * @code
 * arma::mat input;
 * Load("train.csv", input);
 * arma::mat output;
 *
 * // Fit the features.
 * StandardScaler scale;
 * scale.Fit(input)
 *
 * // Scale the features.
 * scale.Transform(input, output);
 *
 * // Retransform the input.
 * scale.InverseTransform(output, input);
 * @endcode
 */
class StandardScaler
{
 public:
  /**
   * Function to fit features, to find out the min max and scale.
   *
   * @param input Dataset to fit.
   */
  template<typename MatType>
  void Fit(const MatType& input)
  {
    itemMean = arma::mean(input, 1);
    itemStdDev = arma::stddev(input, 1, 1);
    // Handle zeros in scale vector.
    itemStdDev.for_each([](arma::vec::elem_type& val) { val =
        (val == 0) ? 1 : val; });
  }

  /**
   * Function to scale features.
   *
   * @param input Dataset to scale features.
   * @param output Output matrix with scaled features.
   */
  template<typename MatType>
  void Transform(const MatType& input, MatType& output)
  {
    if (itemMean.is_empty() || itemStdDev.is_empty())
    {
      throw std::runtime_error("Call Fit() before Transform(), please"
        " refer to the documentation.");
    }
    output.copy_size(input);
    output = (input.each_col() - itemMean).each_col() / itemStdDev;
  }

  /**
   * Function to retrieve original dataset.
   *
   * @param input Scaled dataset.
   * @param output Output matrix with original Dataset.
   */
  template<typename MatType>
  void InverseTransform(const MatType& input, MatType& output)
  {
    output.copy_size(input);
    output = (input.each_col() % itemStdDev).each_col() + itemMean;
  }

  //! Get the mean row vector.
  const arma::vec& ItemMean() const { return itemMean; }
  //! Get the standard deviation row vector.
  const arma::vec& ItemStdDev() const { return itemStdDev; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(itemMean));
    ar(CEREAL_NVP(itemStdDev));
  }

 private:
  // Vector which holds mean of each feature.
  arma::vec itemMean;
  // Vector which holds standard devation of each feature.
  arma::vec itemStdDev;
}; // class StandardScaler

} // namespace data
} // namespace mlpack

#endif
