// Temporarily drop.
/**
 * @file methods/ann/layer/transposed_convolution.hpp
 * @author Shikhar Jaiswal
 * @author Marcus Edel
 *
 * Definition of the Transposed Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_TRANSPOSED_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_LAYER_TRANSPOSED_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/svd_convolution.hpp>
#include <mlpack/core/util/to_lower.hpp>

#include "layer.hpp"
#include "padding.hpp"

namespace mlpack {

/**
 * Implementation of the Transposed Convolution class. The Transposed
 * Convolution class represents a single layer of a neural network.
 *
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class TransposedConvolutionType : public Layer<InputType, OutputType>
{
 public:
  //! Create the Transposed Convolution object.
  TransposedConvolutionType();

  /**
   * Create the Transposed Convolution object using the specified number of
   * input maps, output maps, filter size, stride and padding parameter.
   *
   * Note: The equivalent stride of a transposed convolution operation is always
   * equal to 1. In this implementation, stride of filter represents the stride
   * of the associated convolution operation.
   * Note: Padding of input represents padding of associated convolution
   * operation.
   *
   * @param inSize The number of input maps.
   * @param outSize The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param inputWidth The width of the input data.
   * @param inputHeight The height of the input data.
   * @param outputWidth The width of the output data.
   * @param outputHeight The height of the output data.
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  // TODO: remove inputWidth and inputHeight?
  TransposedConvolutionType(const size_t inSize,
                            const size_t outSize,
                            const size_t kernelWidth,
                            const size_t kernelHeight,
                            const size_t strideWidth = 1,
                            const size_t strideHeight = 1,
                            const size_t padW = 0,
                            const size_t padH = 0,
                            const size_t inputWidth = 0,
                            const size_t inputHeight = 0,
                            const size_t outputWidth = 0,
                            const size_t outputHeight = 0,
                            const std::string& paddingType = "None");

  /**
   * Create the Transposed Convolution object using the specified number of
   * input maps, output maps, filter size, stride and padding parameter.
   *
   * Note: The equivalent stride of a transposed convolution operation is always
   * equal to 1. In this implementation, stride of filter represents the stride
   * of the associated convolution operation.
   * Note: Padding of input represents padding of associated convolution
   * operation.
   *
   * @param inSize The number of input maps.
   * @param outSize The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW A two-value tuple indicating padding widths of the input.
   *             First value is padding at left side. Second value is padding on
   *             right side.
   * @param padH A two-value tuple indicating padding heights of the input.
   *             First value is padding at top. Second value is padding on
   *             bottom.
   * @param inputWidth The width of the input data.
   * @param inputHeight The height of the input data.
   * @param outputWidth The width of the output data.
   * @param outputHeight The height of the output data.
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  TransposedConvolutionType(const size_t inSize,
                            const size_t outSize,
                            const size_t kernelWidth,
                            const size_t kernelHeight,
                            const size_t strideWidth,
                            const size_t strideHeight,
                            const std::tuple<size_t, size_t>& padW,
                            const std::tuple<size_t, size_t>& padH,
                            const size_t inputWidth = 0,
                            const size_t inputHeight = 0,
                            const size_t outputWidth = 0,
                            const size_t outputHeight = 0,
                            const std::string& paddingType = "None");

  /*
   * Set the weight and bias term.
   */
  void SetWeights(const typename OutputType::elem_type* weightsPtr);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param * (input) The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& /* input */,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the weight of the layer.
  arma::Cube<typename OutputType::elem_type> const& Weight() const
  {
    return weight;
  }
  //! Modify the weight of the layer.
  arma::Cube<typename OutputType::elem_type>& Weight() { return weight; }

  //! Get the bias of the layer.
  OutputType const& Bias() const { return bias; }
  //! Modify the bias of the layer.
  OutputType& Bias() { return bias; }

  //! Get the input width.
  size_t const& InputWidth() const { return inputWidth; }
  //! Modify input the width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the input height.
  size_t const& InputHeight() const { return inputHeight; }
  //! Modify the input height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the output width.
  size_t const& OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t const& OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the input size.
  size_t const& InputSize() const { return inSize; }

  //! Get the output size.
  size_t const& OutputSize() const { return outSize; }

  //! Get the kernel width.
  size_t const& KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  size_t const& KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  size_t const& StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  size_t const& StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  //! Get the top padding height.
  size_t const& PadHTop() const { return padHTop; }
  //! Modify the top padding height.
  size_t& PadHTop() { return padHTop; }

  //! Get the bottom padding height.
  size_t const& PadHBottom() const { return padHBottom; }
  //! Modify the bottom padding height.
  size_t& PadHBottom() { return padHBottom; }

  //! Get the left padding width.
  size_t const& PadWLeft() const { return padWLeft; }
  //! Modify the left padding width.
  size_t& PadWLeft() { return padWLeft; }

  //! Get the right padding width.
  size_t const& PadWRight() const { return padWRight; }
  //! Modify the right padding width.
  size_t& PadWRight() { return padWRight; }

  //! Get the size of the weight matrix.
  size_t WeightSize() const
  {
    return (outSize * inSize * kernelWidth * kernelHeight) + outSize;
  }

  const std::vector<size_t>& OutputDimensions() const
  {
    std::vector<size_t> result(inputDimensions.size(), 0);
    result[0] = outputWidth;
    result[1] = outputHeight;
    // Higher dimensions are unmodified.
    for (size_t i = 2; i < inputDimensions.size(); ++i)
      result[i] = inputDimensions[i];
    return result;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /*
   * Rotates a 3rd-order tensor counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  template<typename eT>
  void Rotate180(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = arma::Cube<eT>(input.n_rows, input.n_cols, input.n_slices);

    // * left-right flip, up-down flip */
    for (size_t s = 0; s < output.n_slices; s++)
      output.slice(s) = arma::fliplr(arma::flipud(input.slice(s)));
  }

  /*
   * Function to assign padding such that output size is same as input size.
   */
  void InitializeSamePadding();

  /*
   * Rotates a dense matrix counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  template<typename eT>
  void Rotate180(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    // * left-right flip, up-down flip */
    output = arma::fliplr(arma::flipud(input));
  }


  /*
   * Insert zeros between the units of the given input data.
   * Note: This function should be used before using padding layer.
   *
   * @param input The input to be padded.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param output The padded output data.
   */
  template<typename eT>
  void InsertZeros(const arma::Mat<eT>& input,
                   const size_t strideWidth,
                   const size_t strideHeight,
                   arma::Mat<eT>& output)
  {
    if (output.n_rows != input.n_rows * strideWidth - strideWidth + 1 ||
        output.n_cols != input.n_cols * strideHeight - strideHeight + 1)
    {
      output = zeros(input.n_rows * strideWidth - strideWidth + 1,
          input.n_cols * strideHeight - strideHeight + 1);
    }

    for (size_t i = 0; i < output.n_rows; i += strideHeight)
    {
      for (size_t j = 0; j < output.n_cols; j += strideWidth)
      {
        // TODO: Use [] instead of () for speedup after this is completely
        // debugged and approved.
        output(i, j) = input(i / strideHeight, j / strideWidth);
      }
    }
  }

  /*
   * Insert zeros between the units of the given input data.
   * Note: This function should be used before using padding layer.
   *
   * @param input The input to be padded.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param output The padded output data.
   */
  template<typename eT>
  void InsertZeros(const arma::Cube<eT>& input,
                   const size_t strideWidth,
                   const size_t strideHeight,
                   arma::Cube<eT>& output)
  {
    output = zeros(input.n_rows * strideWidth - strideWidth + 1,
        input.n_cols * strideHeight - strideHeight + 1, input.n_slices);

    for (size_t i = 0; i < input.n_slices; ++i)
    {
      InsertZeros<eT>(input.slice(i), strideWidth, strideHeight,
          output.slice(i));
    }
  }

  //! Locally-stored number of input channels.
  size_t inSize;

  //! Locally-stored number of output channels.
  size_t outSize;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Locally-stored filter/kernel width.
  size_t kernelWidth;

  //! Locally-stored filter/kernel height.
  size_t kernelHeight;

  //! Locally-stored stride of the filter in x-direction.
  size_t strideWidth;

  //! Locally-stored stride of the filter in y-direction.
  size_t strideHeight;

  //! Locally-stored left-side padding width.
  size_t padWLeft;

  //! Locally-stored right-side padding width.
  size_t padWRight;

  //! Locally-stored bottom padding height.
  size_t padHBottom;

  //! Locally-stored top padding height.
  size_t padHTop;

  //! Locally-stored number of zeros added to the right of input.
  size_t aW;

  //! Locally-stored number of zeros added to the bottom of input.
  size_t aH;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored weight object.
  arma::Cube<typename OutputType::elem_type> weight;

  //! Locally-stored bias term object.
  OutputType bias;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored transformed output parameter.
  arma::Cube<typename OutputType::elem_type> outputTemp;

  //! Locally-stored transformed padded input parameter.
  arma::Cube<typename InputType::elem_type> inputPaddedTemp;

  //! Locally-stored transformed expanded input parameter.
  arma::Cube<typename InputType::elem_type> inputExpandedTemp;

  //! Locally-stored transformed error parameter.
  arma::Cube<typename OutputType::elem_type> gTemp;

  //! Locally-stored transformed gradient parameter.
  arma::Cube<typename OutputType::elem_type> gradientTemp;

  //! Locally-stored padding layer for forward propagation.
  Padding paddingForward;

  //! Locally-stored padding layer for back propagation.
  Padding paddingBackward;
}; // class TransposedConvolutionType

// Standard TransposedConvolution
using TransposedConvolution = TransposedConvolutionType<
    NaiveConvolution<ValidConvolution>,
    NaiveConvolution<ValidConvolution>,
    NaiveConvolution<ValidConvolution>,
    arma::mat,
    arma::mat>;

} // namespace mlpack

// Include implementation.
#include "transposed_convolution_impl.hpp"

#endif
