/**
 * @file core/data/scaler_methods/min_max_scaler.hpp
 * @author Jeffin Sam
 *
 * MinMaxScaler class to scale features.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SCALE_HPP
#define MLPACK_CORE_DATA_SCALE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * A simple MinMax Scaler class
 *
 * Given an input dataset this class helps you to scale each
 * feature to a given range.
 *
 * \[z = scale * x + scaleMin - min(x) * scale\]
 * \[scale = (scaleMax - scaleMin) / (max(x) - min(x))\]
 *
 * where scaleMin, scaleMax = feature_range and min(x), max(x)
 * are the minimum and maximum value of x respectively.
 *
 * @code
 * arma::mat input;
 * Load("train.csv", input);
 * arma::mat output;
 *
 * // Fit the features.
 * MinMaxScaler scale;
 * scale.Fit(input)
 *
 * // Scale the features.
 * scale.Transform(input, output);
 *
 * // Retransform the input.
 * scale.InverseTransform(output, input);
 * @endcode
 */
class MinMaxScaler
{
 public:
  /**
   * Default constructor
   *
   * @param min Lower range of scaling.
   * @param max Upper range of scaling.
   */
  MinMaxScaler(const double min = 0, const double max = 1)
  {
    scaleMin = min;
    scaleMax = max;
    // Ensure scaleMin is smaller than scaleMax.
    if (scaleMin > scaleMax)
    {
      throw std::runtime_error("Range is not appropriate");
    }
  }

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
    scale = itemMax - itemMin;
    // Handle zeros in scale vector.
    scale.for_each([](arma::vec::elem_type& val) { val =
        (val == 0) ? 1 : val; });
    scale = (scaleMax - scaleMin) / scale;
    scalerowmin.copy_size(itemMin);
    scalerowmin.fill(scaleMin);
    scalerowmin = scalerowmin - itemMin % scale;
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
    if (scalerowmin.is_empty() || scale.is_empty())
    {
      throw std::runtime_error("Call Fit() before Transform(), please"
          " refer to the documentation.");
    }
    output.copy_size(input);
    output = (input.each_col() % scale).each_col() + scalerowmin;
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
    output = (input.each_col() - scalerowmin).each_col() / scale;
  }

  //! Get the Min row vector.
  const arma::vec& ItemMin() const { return itemMin; }
  //! Get the Max row vector.
  const arma::vec& ItemMax() const { return itemMax; }
  //! Get the Scale row vector.
  const arma::vec& Scale() const { return scale; }
  //! Get the upper range parameter.
  double ScaleMax() const { return scaleMax; }
  //! Get the lower range parameter.
  double ScaleMin() const { return scaleMin; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(itemMin));
    ar(CEREAL_NVP(itemMax));
    ar(CEREAL_NVP(scale));
    ar(CEREAL_NVP(scaleMin));
    ar(CEREAL_NVP(scaleMax));
    ar(CEREAL_NVP(scalerowmin));
  }

 private:
  // Vector which holds minimum of each feature.
  arma::vec itemMin;
  // Vector which holds maximum of each feature.
  arma::vec itemMax;
  // Scale vector which is used to scale up each feature.
  arma::vec scale;
  // Lower value for range.
  double scaleMin;
  // Upper value for range.
  double scaleMax;
  // Column vector of scalemin
  arma::vec scalerowmin;
}; // class MinMaxScaler

} // namespace data
} // namespace mlpack

#endif
