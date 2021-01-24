/**
 * @file methods/ann/layer/adaptive_max_pooling.hpp
 * @author Kartik Dutt
 *
 * Definition of the AdaptiveMaxPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"
#include "max_pooling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the AdaptiveMaxPooling layer.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *    cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the layer's Outputs. The layer automatically
 *    cast inputs to this type (Default: arma::mat).
 */
template <typename InputType = arma::mat, typename OutputType = arma::mat>
class AdaptiveMaxPoolingType : public Layer<InputType, OutputType>
{
 public:
  //! Create the AdaptiveMaxPooling object.
  AdaptiveMaxPoolingType();

  /**
   * Create the AdaptiveMaxPooling object.
   *
   * @param outputWidth Width of the output.
   * @param outputHeight Height of the output.
   */
  AdaptiveMaxPoolingType(const size_t outputWidth,
                         const size_t outputHeight);

  /**
   * Create the AdaptiveMaxPooling object.
   *
   * @param outputShape A two-value tuple indicating width and height of the output.
   */
  AdaptiveMaxPoolingType(const std::tuple<size_t, size_t>& outputShape);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g);

  //! Get the input width.
  size_t const& InputWidth() const { return poolingLayer.InputWidth(); }
  //! Modify the input width.
  size_t& InputWidth() { return poolingLayer.InputWidth(); }

  //! Get the input height.
  size_t const& InputHeight() const { return poolingLayer.InputHeight(); }
  //! Modify the input height.
  size_t& InputHeight() { return poolingLayer.InputHeight(); }

  //! Get the output width.
  size_t const& OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t const& OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the input size.
  size_t const& InputSize() const { return poolingLayer.InputSize(); }

  //! Get the output size.
  size_t OutputSize() const { return poolingLayer.OutputSize(); }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  /**
   * Initialize Kernel Size and Stride for Adaptive Pooling.
   */
  void IntializeAdaptivePadding()
  {
    poolingLayer.StrideWidth() = std::floor(poolingLayer.InputWidth() /
        outputWidth);
    poolingLayer.StrideHeight() = std::floor(poolingLayer.InputHeight() /
        outputHeight);

    poolingLayer.KernelWidth() = poolingLayer.InputWidth() -
        (outputWidth - 1) * poolingLayer.StrideWidth();
    poolingLayer.KernelHeight() = poolingLayer.InputHeight() -
        (outputHeight - 1) * poolingLayer.StrideHeight();

    if (poolingLayer.KernelHeight() <= 0 || poolingLayer.KernelWidth() <= 0 ||
        poolingLayer.StrideWidth() <= 0 || poolingLayer.StrideHeight() <= 0)
    {
      Log::Fatal << "Given output shape (" << outputWidth << ", "
        << outputHeight << ") is not possible for given input shape ("
        << poolingLayer.InputWidth() << ", " << poolingLayer.InputHeight()
        << ")." << std::endl;
    }
  }

  //! Locally stored MaxPooling Object.
  MaxPoolingType<InputType, OutputType> poolingLayer;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored reset parameter used to initialize the layer once.
  bool reset;
}; // class AdaptiveMaxPoolingType

// Convenience typedefs.

// Standard Adaptive max pooling layer.
typedef AdaptiveMaxPoolingType<arma::mat, arma::mat> AdaptiveMaxPooling;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "adaptive_max_pooling_impl.hpp"

#endif
