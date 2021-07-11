/**
 * @file methods/ann/layer/unpooling.hpp
 * @author Anjishnu Mukherjee
 *
 * Definition of the UnPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_UNPOOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_UNPOOLING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class UnPooling
{
 public:
  /**
   * Create the UnPooling object using the specified number of units.
   *
   * @param poolingIndices The indices of the max values, obtained from maxpool.
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   */
  UnPooling(std::vector<arma::cube> poolingIndices,
            const size_t kernelWidth,
            const size_t kernelHeight,
            const size_t strideWidth = 1,
            const size_t strideHeight = 1);

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
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  const OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  const OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the input width.
  size_t InputWidth() const { return inputWidth; }
  //! Modify the input width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the input height.
  size_t InputHeight() const { return inputHeight; }
  //! Modify the input height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the output width.
  size_t OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the input size.
  size_t InputSize() const { return inSize; }

  //! Get the output size.
  size_t OutputSize() const { return outSize; }

  //! Get the kernel width.
  size_t KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  size_t KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  size_t StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  size_t StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Apply unpooling to the input and store the results.
   *
   * @param input A slice from inputTemp.
   * @param output Corresponding slice from outputTemp.
   * @param poolingIndices The pooled indices.
   */
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& input,
                 arma::Mat<eT>& output,
                 arma::Mat<eT>& poolingIndices)
  {
    for (size_t i = 0; i < poolingIndices.n_elem; ++i)
    {
      output(poolingIndices(i)) += input(i);
    }
  }

  //! Locally-stored width of the pooling window.
  size_t kernelWidth;

  //! Locally-stored height of the pooling window.
  size_t kernelHeight;

  //! Locally-stored width of the stride operation.
  size_t strideWidth;

  //! Locally-stored height of the stride operation.
  size_t strideHeight;

  //! Locally-stored number of input channels.
  size_t inSize;

  //! Locally-stored number of output channels.
  size_t outSize;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored pooling indicies.
  std::vector<arma::cube> poolingIndices;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed output parameter.
  arma::cube gTemp;
}; // class UnPooling

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "unpooling_impl.hpp"

#endif
