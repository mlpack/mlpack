/**
 * @file adaptive_max_pooling.hpp
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
  AdaptiveMaxPooling(const std::tuple<size_t, size_t> outputShape);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);

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
  void Backward(const arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  //! Get the output parameter.
  const OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  const OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the width.
  size_t InputWidth() const { return inputWidth; }
  //! Modify the width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the height.
  size_t InputHeight() const { return inputHeight; }
  //! Modify the height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the width.
  size_t OutputWidth() const { return outputWidth; }
  //! Modify the width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the height.
  size_t OutputHeight() const { return outputHeight; }
  //! Modify the height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the input size.
  size_t InputSize() const { return inSize; }

  //! Get the output size.
  size_t OutputSize() const { return outSize; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Initialize Kernel Size and Stride for Adaptive Pooling.
   */
  void IntializeAdaptivePadding()
  {
    strideWidth = std::floor(inputWidth / outputWidth);
    strideHeight = std::floor(inputHeight / outputHeight);

    kernelWidth = inputWidth - (outputWidth - 1) * strideWidth;
    kernelHeight = inputHeight - (outputHeight - 1) * strideHeight;

    if (kernelHeight < 0 || kernelWidth < 0)
    {
      Log::Fatal << "Given output shape (" << outputWidth << ", "
        << outputHeight << ") is not possible for given input shape ("
        << inputWidth <<", "<< inputHeight << ")."<< std::endl;
    }
  }

  /**
   * Apply pooling to the input and store the results.
   *
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   * @param poolingIndices The pooled indices.
   */
  template<typename eT>
  void PoolingOperation(const arma::Mat<eT>& input,
                        arma::Mat<eT>& output,
                        arma::Mat<eT>& poolingIndices)
  {
    const size_t rStep = kernelWidth;
    const size_t cStep = kernelHeight;
    for (size_t j = 0, colidx = 0; j < output.n_cols;
         ++j, colidx += strideHeight)
    {
      for (size_t i = 0, rowidx = 0; i < output.n_rows;
           ++i, rowidx += strideWidth)
      {
        arma::mat subInput = input(
            arma::span(rowidx, rowidx + rStep - 1),
            arma::span(colidx, colidx + cStep - 1));
        const size_t idx = pooling.Pooling(subInput);
        output(i, j) = subInput(idx);
        arma::Mat<size_t> subIndices = indices(arma::span(rowidx,
              rowidx + rStep - 1),
              arma::span(colidx, colidx + cStep - 1));
        poolingIndices(i, j) = subIndices(idx);
      }
    }
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param error The backward error.
   * @param output The pooled result.
   * @param poolingIndices The pooled indices.
   */
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& error,
                 arma::Mat<eT>& output,
                 arma::Mat<eT>& poolingIndices)
  {
    for (size_t i = 0; i < poolingIndices.n_elem; ++i)
    {
      output(poolingIndices(i)) += error(i);
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

  //! Locally-stored reset parameter used to initialize the module once.
  bool reset;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Locally-stored output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed output parameter.
  arma::cube gTemp;

  //! Locally-stored pooling strategy.
  MaxPoolingRule pooling;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored indices matrix parameter.
  arma::Mat<size_t> indices;

  //! Locally-stored indices column parameter.
  arma::Col<size_t> indicesCol;

  //! Locally-stored pooling indicies.
  std::vector<arma::cube> poolingIndices;
}; // class MaxPooling

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "adaptive_max_pooling_impl.hpp"

#endif
