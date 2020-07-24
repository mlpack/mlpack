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
#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the AdaptiveMaxPooling layer.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class AdaptiveMaxPooling
{
 public:
  //! Create the AdaptiveMaxPooling object.
  AdaptiveMaxPooling();

  /**
   * Create the AdaptiveMaxPooling object.
   *
   * @param outputWidth Width of the output.
   * @param outputHeight Height of the output.
   */
  AdaptiveMaxPooling(const size_t outputWidth,
                     const size_t outputHeight);

  /**
   * Create the AdaptiveMaxPooling object.
   *
   * @param outputShape A two-value tuple indicating width and height of the output.
   */
  AdaptiveMaxPooling(const std::tuple<size_t, size_t>& outputShape);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  const OutputDataType& OutputParameter() const
  { return poolingLayer.OutputParameter(); }

  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return poolingLayer.OutputParameter(); }

  //! Get the delta.
  const OutputDataType& Delta() const { return poolingLayer.Delta(); }
  //! Modify the delta.
  OutputDataType& Delta() { return poolingLayer.Delta(); }

  //! Get the input width.
  size_t InputWidth() const { return poolingLayer.InputWidth(); }
  //! Modify the input width.
  size_t& InputWidth() { return poolingLayer.InputWidth(); }

  //! Get the input height.
  size_t InputHeight() const { return poolingLayer.InputHeight(); }
  //! Modify the input height.
  size_t& InputHeight() { return poolingLayer.InputHeight(); }

  //! Get the output width.
  size_t OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the input size.
  size_t InputSize() const { return poolingLayer.InputSize(); }

  //! Get the output size.
  size_t OutputSize() const { return poolingLayer.OutputSize(); }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar);

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
  MaxPooling<InputDataType, OutputDataType> poolingLayer;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored reset parameter used to initialize the layer once.
  bool reset;
}; // class AdaptiveMaxPooling

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "adaptive_max_pooling_impl.hpp"

#endif
