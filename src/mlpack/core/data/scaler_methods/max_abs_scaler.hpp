/**
 * @file core/data/scaler_methods/max_abs_scaler.hpp
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
 * \[z = x / max(abs(x))\]
 *
 * where max(abs(x)) is maximum absolute value of feature.
 *
 * @code
 * arma::mat input;
 * Load("train.csv", input);
 * arma::mat output;
 *
 * // Fit the features.
 * MaxAbsScaler scale;
 * scale.Fit(input)
 *
 * // Scale the features.
 * scale.Transform(input, output);
 *
 * // Retransform the input.
 * scale.InverseTransform(output, input);
 * @endcode
 */
class MaxAbsScaler
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
    itemMin = min(input, 1);
    itemMax = arma::max(input, 1);
    scale = arma::max(arma::abs(itemMin), arma::abs(itemMax));
    // Handling zeros in scale vector.
    scale.for_each([](arma::vec::elem_type& val) { val =
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
    if (scale.is_empty())
    {
      throw std::runtime_error("Call Fit() before Transform(), please"
        " refer to the documentation.");
    }
    output.copy_size(input);
    output = input.each_col() / scale;
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
    output = input.each_col() % scale;
  }

  //! Get the Min row vector.
  const arma::vec& ItemMin() const { return itemMin; }
  //! Get the Max row vector.
  const arma::vec& ItemMax() const { return itemMax; }
  //! Get the Scale row vector.
  const arma::vec& Scale() const { return scale; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(itemMin));
    ar(CEREAL_NVP(itemMax));
    ar(CEREAL_NVP(scale));
  }
 private:
  // Vector which holds minimum of each feature.
  arma::vec itemMin;
  // Vector which holds maximum of each feature.
  arma::vec itemMax;
  // Vector which is used to scale up each feature.
  arma::vec scale;
}; // class MaxAbsScaler

} // namespace data
} // namespace mlpack

#endif
