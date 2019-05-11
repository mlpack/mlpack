/**
 * @file minmaxscaler.hpp
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
 * @code
 * arma::Mat input = loadData();
 * arma::Mat output;
 *
 * // Scale the features.
 * MinMaxScaler scale;
 * scale.Tranform(input, output);
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
    scalemin = min;
    scalemax = max;
  }

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
    itemMin = arma::min(input, 1);
    itemMax = arma::max(input, 1);
    scale = itemMax - itemMin;
    // Handling zeros in scale vector.
    for (size_t i = 0; i < scale.n_elem; i++)
    {
      if (scale(i) == 0)
      {
        scale(i) = 1;
      }
    }
    scale = (scalemax - scalemin) / scale;
    scalerowmin.copy_size(itemMin);
    scalerowmin.fill(scalemin);
    scalerowmin = scalerowmin - itemMin % scale;
    output = (input.each_col() % scale).each_col() + scalerowmin; 
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
    output = (input.each_col() - scalerowmin).each_col() % (1.0 / scale);
  }

  //! Get the Min row vector.
  const arma::colvec& ItemMin() const { return itemMin; }
  //! Get the Max row vector.
  const arma::colvec& ItemMax() const { return itemMax; }
  //! Get the Scale row vector.
  const arma::colvec& Scale() const { return scale; }
  //! Get the upper range parameter.
  const double ScaleMax() const { return scalemax; }
  //! Get the lower range parameter.
  const double ScaleMin() const { return scalemin; }

 private:
  // Vector which holds minimum of each feature.
  arma::colvec itemMin;
  // Vector which holds maximum of each feature.
  arma::colvec itemMax;
  // Scale vector which is used to scale up each feature.
  arma::colvec scale;
  // Lower value for range.
  double scalemin;
  // Upper value for range.
  double scalemax;
  // Column vector of scalemin
  arma::colvec scalerowmin;
}; // class MinMaxScaler

} // namespace data
} // namespace mlpack

#endif
