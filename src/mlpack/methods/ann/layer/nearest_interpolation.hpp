/**
 * @file methods/ann/layer/nearest_interpolation.hpp
 * @author Abhinav Anand
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

  NearestInterpolationType(const size_t inRowSize,
                           const size_t inColSize,
                           const size_t outRowSize,
                           const size_t outColSize,
                           const size_t depth);

  NearestInterpolationType* Clone() const { 
    return new NearestInterpolationType(*this);
  }

  virtual ~NearestInterpolationType() { }

  //! Copy the given ConcatenateType layer.
  NearestInterpolationType(const NearestInterpolationType& other);
  //! Take ownership of the given ConcatenateType layer.
  NearestInterpolationType(NearestInterpolationType&& other);
  //! Copy the given ConcatenateType layer.
  NearestInterpolationType& operator=(const NearestInterpolationType& other);
  //! Take ownership of the given ConcatenateType layer.
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

  void ComputeOutputDimensions();
  //! Get the row size of the input.
  size_t const& InRowSize() const { return inRowSize; }
  //! Modify the row size of the input.
  size_t& InRowSize() { return inRowSize; }
  
  //! Get the column size of the input.
  size_t const& InColSize() const { return inColSize; }
  //! Modify the column size of the input.
  size_t& InColSize() { return inColSize; }
  
  //! Get the row size of the output.
  size_t const& OutRowSize() const { return outRowSize; }
  //! Modify the row size of the output.
  size_t& OutRowSize() { return outRowSize; }
  
  //! Get the column size of the output.
  size_t const& OutColSize() const { return outColSize; }
  //! Modify the column size of the output.
  size_t& OutColSize() { return outColSize; }
  
  //! Get the depth of the input.
  size_t const& InDepth() const { return depth; }
  //! Modify the depth of the input.
  size_t& InDepth() { return depth; }
  
  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored row size of the input.
  size_t inRowSize;
  //! Locally stored column size of the input.
  size_t inColSize;
  //! Locally stored row size of the output.
  size_t outRowSize;
  //! Locally stored column size of the input.
  size_t outColSize;
  //! Locally stored depth of the input.
  size_t depth;
  //! Locally stored number of input points.
  size_t batchSize;
}; // class NearestInterpolation

typedef NearestInterpolationType<arma::mat> NearestInterpolation;

} // namespace mlpack

// Include implementation.
#include "nearest_interpolation_impl.hpp"

#endif
