//
/**
 * @filer methods/ann/layer/nearest_interpolation.hpp
 * @author Andrew Furey
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEAREST_INTERPOLATION_HPP
#define MLPACK_METHODS_ANN_LAYER_NEAREST_INTERPOLATION_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * Definition and Implementation of the Nearest Interpolation Layer.
 *
 * Nearest interpolation is an mathematical technique, primarily used for
 * scaling purposes. The input should be a 2D matrix and it can have
 * a number of channels/units.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam MatType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<typename MatType = arma::mat>
class NearestInterpolationType : public Layer<MatType>
{
 public:
  //! Create the NearestInterpolation object.
  NearestInterpolationType();

  /**Create NearestInterpolation Object with the same scaleFactor along 
   * each dimension.
   * NOTE: scaleFactors must be a two element vector, the first element
   * for scaling the first dimension and the second element for scaling
   * the second dimension.
   *
   * If the input dimensions are n x m x ..., then the output dimensions
   * will be (n x scaleFactors[0]) x (m x scaleFactors[1]) x ...
   * 
   * @param scaleFactor Scale factors to scale each dimension by.
   */
  NearestInterpolationType(const std::vector<double> scaleFactors);

  NearestInterpolationType* Clone() const {
    return new NearestInterpolationType(*this);
  }

  virtual ~NearestInterpolationType() { }

  //! Copy the given NearestInterpolationType layer.
  NearestInterpolationType(const NearestInterpolationType& other);
  //! Take ownership of the given NearestInterpolationType layer.
  NearestInterpolationType(NearestInterpolationType&& other);
  //! Copy the given NearestInterpolationType layer.
  NearestInterpolationType& operator=(const NearestInterpolationType& other);
  //! Take ownership of the given NearestInterpolationType layer.
  NearestInterpolationType& operator=(NearestInterpolationType&& other);

  /**
   * Forward pass through the layer. The layer interpolates
   * the matrix using the given Nearest Interpolation method.
   *
   * @param input The input matrix.
   * @param output The resulting interpolated output matrix.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass. Since the layer does not have any learn-able parameters,
   * we just have to down-sample the gradient to make its size compatible with
   * the input size.
   *
   * @param * (input) The input matrix.
   * @param gradient The computed backward gradient.
   * @param output The resulting down-sampled output.
   */
  void Backward(const MatType& /*input*/,
                const MatType& gradient,
                MatType& output);

  //! Compute the output dimensions of the layer, based on the internal values
  //! of `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Vector of scale factors to scale different dimensions.
  std::vector<double> scaleFactors;
}; // class NearestInterpolation

using NearestInterpolation = NearestInterpolationType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "nearest_interpolation_impl.hpp"

#endif
