// Maybe 
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
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MeanPooling.
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

  /**
   * Create the AdaptiveMeanPooling object.
   *
   * @param outputShape A two-value tuple indicating width and height of the
   *      output.
   */
  AdaptiveMeanPoolingType(const std::tuple<size_t, size_t>& outputShape);

  // Virtual destructor.
  virtual ~AdaptiveMeanPoolingType() { 
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

  AdaptiveMeanPoolingType* Clone() const { return new AdaptiveMeanPoolingType(*this); }

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
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
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
  /**
   * Initialize Kernel Size and Stride for Adaptive Pooling.
   */
  void InitializeAdaptivePadding()
  {
    poolingLayer.InputDimensions() = this->inputDimensions;
    poolingLayer.StrideWidth() = std::floor(this->inputDimensions[0] /
        outputWidth);
    poolingLayer.StrideHeight() = std::floor(this->inputDimensions[1] /
        outputHeight);

    poolingLayer.KernelWidth() = this->inputDimensions[0] -
        (outputWidth - 1) * poolingLayer.StrideWidth();
    poolingLayer.KernelHeight() = this->inputDimensions[1] -
        (outputHeight - 1) * poolingLayer.StrideHeight();

    if (poolingLayer.KernelHeight() <= 0 || poolingLayer.KernelWidth() <= 0 ||
        poolingLayer.StrideWidth() <= 0 || poolingLayer.StrideHeight() <= 0)
    {
      Log::Fatal << "Given output shape (" << outputWidth << ", "
        << outputHeight << ") is not possible for given input shape ("
        << this->inputDimensions[0] << ", " << this->inputDimensions[1]
        << ")." << std::endl;
    }
    poolingLayer.ComputeOutputDimensions();
  }

  //! Locally stored MeanPooling Object.
  MeanPoolingType<MatType> poolingLayer;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;
}; // class AdaptiveMeanPoolingType

// Convenience typedefs.

// Standard Adaptive mean pooling layer.
typedef AdaptiveMeanPoolingType<arma::mat> AdaptiveMeanPooling;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "adaptive_mean_pooling_impl.hpp"

#endif
