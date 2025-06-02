/**
 * @file methods/ann/layer/adaptive_mean_pooling.hpp
 * @author Kartik Dutt
 *
 * Definition of the AdaptiveMeanPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MEAN_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MEAN_POOLING_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"
#include "mean_pooling.hpp"

namespace mlpack {

/**
 * Implementation of the AdaptiveMeanPooling layer.
 *
 * The AdaptiveMeanPooling layer works similarly to MeanPooling layer, but it
 * adaptively changes the size of the pooling region to minimize the amount of
 * computation. In MeanPooling, we specifies the kernel and stride size whereas
 * in AdaptiveMeanPooling, we specify the output size of the pooling region.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *         computation.
 */
template <typename MatType = arma::mat>
class AdaptiveMeanPoolingType : public Layer<MatType>
{
 public:
  //! Create the AdaptiveMeanPooling object.
  AdaptiveMeanPoolingType();

  /**
   * Create the AdaptiveMeanPooling object.
   *
   * @param outputWidth Width of the output.
   * @param outputHeight Height of the output.
   */
  AdaptiveMeanPoolingType(const size_t outputWidth,
                          const size_t outputHeight);

  // Virtual destructor.
  virtual ~AdaptiveMeanPoolingType()
  {
    // Nothing to do here.
  }

  //! Copy the given AdaptiveMeanPoolingType.
  AdaptiveMeanPoolingType(const AdaptiveMeanPoolingType& other);
  //! Take ownership of the given AdaptiveMeanPoolingType.
  AdaptiveMeanPoolingType(AdaptiveMeanPoolingType&& other);
  //! Copy the given AdaptiveMeanPoolingType.
  AdaptiveMeanPoolingType& operator=(const AdaptiveMeanPoolingType& other);
  //! Take ownership of the given AdaptiveMeanPoolingType.
  AdaptiveMeanPoolingType& operator=(AdaptiveMeanPoolingType&& other);

  //! Clone the AdaptiveMeanPoolingType object.
  //! This handles polymorphism correctly.
  AdaptiveMeanPoolingType* Clone() const
  {
    return new AdaptiveMeanPoolingType(*this);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g);

  //! Get the output width.
  size_t const& OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t const& OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Compute the size of the output given `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! Locally stored MeanPooling Object.
  MeanPoolingType<MatType> poolingLayer;

  //! Locally-stored output width. These are user specified outputWidth.
  //! Actual outputWidth will be equal to this but only after
  //! `ComputeOutputDimensions()` is called.
  size_t outputWidth;

  //! Locally-stored output height. These are user specified outputWidth.
  //! Actual outputWidth will be equal to this but only after
  //! `ComputeOutputDimensions()` is called.
  size_t outputHeight;
}; // class AdaptiveMeanPoolingType

// Convenience typedefs.

// Standard Adaptive mean pooling layer.
using AdaptiveMeanPooling = AdaptiveMeanPoolingType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "adaptive_mean_pooling_impl.hpp"

#endif
