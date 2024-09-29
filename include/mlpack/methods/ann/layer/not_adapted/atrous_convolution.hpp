// Temporarily drop.
/**
 * @file methods/ann/layer/atrous_convolution.hpp
 * @author Aarush Gupta
 * @author Shikhar Jaiswal
 *
 * Definition of the Atrous Convolution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/svd_convolution.hpp>
#include <mlpack/core/util/to_lower.hpp>

#include "layer_types.hpp"
#include "padding.hpp"

namespace mlpack{

/**
 * Implementation of the Atrous Convolution class. The Atrous Convolution
 * class represents a single layer of a neural network. Atrous (or Dilated)
 * Convolutions are just simple convolutions applied to input with the defined,
 * spaces included between the kernel cells, in order to capture a larger
 * field of reception, without having to increase dicrete kernel sizes.
 *
 * @tparam ForwardConvolutionRule Atrous Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Atrous Convolution to perform backward process.
 * @tparam GradientConvolutionRule Atrous Convolution to calculate gradient.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class AtrousConvolution : public Layer<InputType, OutputType>
{
 public:
  //! Create the AtrousConvolution object.
  AtrousConvolution();

  /**
   * Create the AtrousConvolution object using the specified number of
   * input maps, output maps, filter size, stride, dilation and
   * padding parameter.
   *
   * @param inSize The number of input maps.
   * @param outSize The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param inputWidth The widht of the input data.
   * @param inputHeight The height of the input data.
   * @param dilationWidth The space between the cells of filters in x direction.
   * @param dilationHeight The space between the cells of filters in y
   *      direction.
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  AtrousConvolution(const size_t inSize,
                    const size_t outSize,
                    const size_t kernelWidth,
                    const size_t kernelHeight,
                    const size_t strideWidth = 1,
                    const size_t strideHeight = 1,
                    const size_t padW = 0,
                    const size_t padH = 0,
                    const size_t inputWidth = 0,
                    const size_t inputHeight = 0,
                    const size_t dilationWidth = 1,
                    const size_t dilationHeight = 1,
                    const std::string& paddingType = "None");

  /**
   * Create the AtrousConvolution object using the specified number of
   * input maps, output maps, filter size, stride, dilation and
   * padding parameter.
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
   * @param inputWidth The widht of the input data.
   * @param inputHeight The height of the input data.
   * @param dilationWidth The space between the cells of filters in x direction.
   * @param dilationHeight The space between the cells of filters in y
   *      direction.
   * @param paddingType The type of padding (Valid/Same/None). Defaults to None.
   */
  AtrousConvolution(const size_t inSize,
                    const size_t outSize,
                    const size_t kernelWidth,
                    const size_t kernelHeight,
                    const size_t strideWidth,
                    const size_t strideHeight,
                    const std::tuple<size_t, size_t>& padW,
                    const std::tuple<size_t, size_t>& padH,
                    const size_t inputWidth = 0,
                    const size_t inputHeight = 0,
                    const size_t dilationWidth = 1,
                    const size_t dilationHeight = 1,
                    const std::string& paddingType = "None");

  /*
   * Set the weight and bias term.
   */
  void Reset();

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

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
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
  const arma::Cube<typename OutputType::elem_type>& Weight() const
  {
    return weight;
  }
  //! Modify the weight of the layer.
  arma::Cube<typename OutputType::elem_type>& Weight() { return weight; }

  const std::vector<size_t>& OutputDimensions() const
  {
    std::vector<size_t> result(inputDimensions.size(), 0);
    result[0] = outputWidth;
    result[1] = outputHeight;
    return result;
  }

  //! Get the bias of the layer.
  const OutputType& Bias() const { return bias; }
  //! Modify the bias of the layer.
  OutputType& Bias() { return bias; }

  //! Get the input width.
  const size_t& InputWidth() const { return inputWidth; }
  //! Modify input the width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the input height.
  const size_t& InputHeight() const { return inputHeight; }
  //! Modify the input height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the output width.
  const size_t& OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  const size_t& OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the kernel width.
  const size_t& KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  const size_t& KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  const size_t& StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  const size_t& StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  //! Get the dilation rate on the X axis.
  const size_t& DilationWidth() const { return dilationWidth; }
  //! Modify the dilation rate on the X axis.
  size_t& DilationWidth() { return dilationWidth; }

  //! Get the dilation rate on the Y axis.
  const size_t& DilationHeight() const { return dilationHeight; }
  //! Modify the dilation rate on the Y axis.
  size_t& DilationHeight() { return dilationHeight; }

  //! Get the internal Padding layer.
  PaddingType<InputType, OutputType> const& Padding() const { return padding; }
  //! Modify the internal Padding layer.
  PaddingType<InputType, OutputType>& Padding() { return padding; }

  //! Get size of the weight matrix.
  size_t WeightSize() const
  {
    return (outSize * inSize * kernelWidth * kernelHeight) + outSize;
  }

  //! Get the shape of the input.
  size_t InputShape() const
  {
    return inputHeight * inputWidth * inSize;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /*
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param pSideOne The size of the padding (width or height) on one side.
   * @param pSideTwo The size of the padding (width or height) on another side.
   * @param d The dilation size.
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t pSideOne,
                     const size_t pSideTwo,
                     const size_t d)
  {
    return std::floor(size + pSideOne + pSideTwo - d * (k - 1) - 1) / s + 1;
  }

  /*
   * Function to assign padding such that output size is same as input size.
   */
  void InitializeSamePadding(size_t& padWLeft,
                             size_t& padWRight,
                             size_t& padHBottom,
                             size_t& padHTop) const;

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

  //! Locally-stored width dilation factor.
  size_t dilationWidth;

  //! Locally-stored height dilation factor.
  size_t dilationHeight;

  //! Locally-stored transformed output parameter.
  arma::Cube<typename OutputType::elem_type> outputTemp;

  //! Locally-stored transformed padded input parameter.
  arma::Cube<typename OutputType::elem_type> inputPaddedTemp;

  //! Locally-stored transformed error parameter.
  arma::Cube<typename OutputType::elem_type> gTemp;

  //! Locally-stored transformed gradient parameter.
  arma::Cube<typename OutputType::elem_type> gradientTemp;

  //! Locally-stored padding layer.
  PaddingType<InputType, OutputType> padding;
}; // class AtrousConvolution

} // namespace mlpack

// Include implementation.
#include "atrous_convolution_impl.hpp"

#endif
