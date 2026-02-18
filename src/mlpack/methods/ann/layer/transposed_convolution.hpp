/**
 * @file methods/ann/layer/transposed_convolution.hpp
 * @author Shikhar Jaiswal
 * @author Marcus Edel
 * @author Ranjodh Singh
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

#include <mlpack/core/util/to_lower.hpp>
#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/im2col_convolution.hpp>

#include "layer.hpp"
#include "padding.hpp"

namespace mlpack
{

/**
 * Implementation of the Transposed Convolution class. The Transposed
 * Convolution class represents a single layer of a neural network.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 */
template <
    typename MatType = arma::mat,
    typename ForwardConvolutionRule = Im2ColConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = Im2ColConvolution<ValidConvolution>,
    typename GradientConvolutionRule = Im2ColConvolution<ValidConvolution>
>
class TransposedConvolution : public Layer<MatType>
{
 public:
  using CubeType = typename GetCubeType<MatType>::type;

  //! Create the Transposed Convolution object.
  TransposedConvolution();

  /**
   * Create the TransposedConvolution object using the specified number of
   * output maps, filter size, stride and padding parameter.
   *
   * Note: The equivalent stride of a transposed convolution operation is always
   * equal to 1. In this implementation, stride of filter represents the stride
   * of the associated convolution operation.
   * Note: Padding of input represents padding of associated convolution
   * operation.
   *
   * @param maps The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param OutputPadW Output padding added to the right side of the input.
   * @param OutputPadH Output padding added to the bottom side of the input.
   * @param paddingType The type of padding ("valid", "same" or "none").
   *    valid: applies no padding.
   *    same: applies padding to keep output size equal to input.
   *    none: applies the padding specified by `padW` and `padH`. (Default)
   * @param useBias Whether or not to use a bias with the convolution.
   */
  TransposedConvolution(const size_t maps,
                        const size_t kernelWidth,
                        const size_t kernelHeight,
                        const size_t strideWidth = 1,
                        const size_t strideHeight = 1,
                        const size_t padW = 0,
                        const size_t padH = 0,
                        const size_t outputPadW = 0,
                        const size_t outputPadH = 0,
                        const std::string& paddingType = "none",
                        const bool useBias = true);

  /**
   * Create the TransposedConvolution object using the specified number of
   * output maps, filter size, stride and padding parameter.
   *
   * Note: The equivalent stride of a transposed convolution operation is always
   * equal to 1. In this implementation, stride of filter represents the stride
   * of the associated convolution operation.
   * Note: Padding of input represents padding of associated convolution
   * operation.
   *
   * @param maps The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW A two-value tuple indicating padding widths of the input.
   *   First value is padding at left side. Second value is padding on
   *   right side.
   * @param padH A two-value tuple indicating padding heights of the input.
   *   First value is padding at top. Second value is padding on
   *   bottom.
   * @param OutputPadW Output padding added to the right side of the input.
   * @param OutputPadH Output padding added to the bottom side of the input.
   * @param paddingType The type of padding ("valid", "same" or "none").
   *   valid: applies no padding.
   *   same: applies padding to keep output size equal to input.
   *   none: applies the padding specified by `padW` and `padH`. (Default)
   * @param useBias Whether or not to use a bias with the convolution.
   */
  TransposedConvolution(const size_t maps,
                        const size_t kernelWidth,
                        const size_t kernelHeight,
                        const size_t strideWidth,
                        const size_t strideHeight,
                        const std::tuple<size_t, size_t>& padW,
                        const std::tuple<size_t, size_t>& padH,
                        const size_t outputPadW = 0,
                        const size_t outputPadH = 0,
                        const std::string& paddingType = "none",
                        const bool useBias = true);

  //! Clone the TransposedConvolution object.
  //! This handles polymorphism correctly.
  TransposedConvolution *Clone() const
  {
    return new TransposedConvolution(*this);
  }

  //! Copy the given TransposedConvolution (but not weights).
  TransposedConvolution(const TransposedConvolution& layer);

  //! Take ownership of the given TransposedConvolution (but not weights).
  TransposedConvolution(TransposedConvolution&& layer);

  //! Copy the given TransposedConvolution (but not weights).
  TransposedConvolution& operator=(const TransposedConvolution& layer);

  //! Take ownership of the given TransposedConvolution (but not weights).
  TransposedConvolution& operator=(TransposedConvolution&& layer);

  //! Virtual destructor.
  virtual ~TransposedConvolution() {}

  /**
   * Set the weight and bias term.
   */
  void SetWeights(const MatType& weightsIn);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& /* input */,
                const MatType& error,
                MatType& gradient);

  //! Get the parameters.
  MatType const& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the weight of the layer as a cube.
  CubeType const& Weight() const { return weight; }
  //! Modify the weight of the layer as a cube.
  CubeType& Weight() { return weight; }

  //! Get the bias of the layer.
  MatType const& Bias() const { return bias; }
  //! Modify the bias of the layer.
  MatType& Bias() { return bias; }

  //! Get the number of output maps.
  size_t const& Maps() const { return maps; }

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

  //! Get size of weights for the layer.
  size_t WeightSize() const
  {
    return (maps * inMaps * kernelWidth * kernelHeight) + (useBias ? maps : 0);
  }

  //! Compute the output dimensions of the layer based on `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template <typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Return the transposed convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param pSideOne The size of the padding (width or height) on one side.
   * @param pSideTwo The size of the padding (width or height) on another side.
   * @return The transposed convolution output size.
   */
  size_t TConvOutSize(const size_t size,
                      const size_t k,
                      const size_t s,
                      const size_t pSideOne,
                      const size_t pSideTwo,
                      const size_t outputPad)
  {
    return (s * (size - 1) + k - pSideOne - pSideTwo + outputPad);
  }

  /**
   * Function to assign padding such that output size is same as input size.
   */
  void InitializeSamePadding();

  /**
   * Rotates a 3rd-order tensor counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  void Rotate180(const CubeType& input, CubeType& output)
  {
    output.set_size(input.n_rows, input.n_cols, input.n_slices);

    // left-right flip, up-down flip
    for (size_t s = 0; s < output.n_slices; s++)
      output.slice(s) = fliplr(flipud(input.slice(s)));
  }

  /**
   * Rotates a dense matrix counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  void Rotate180(const MatType& input, MatType& output)
  {
    // left-right flip, up-down flip
    output = fliplr(flipud(input));
  }

  /**
   * Insert zeros between the units of the given input data.
   * Note: This function should be used before using padding layer.
   *
   * @param input The input to be padded.
   * @param output The padded output data.
   */
  void InsertZeros(const CubeType& input, CubeType& output)
  {
    using UVec = typename GetUColType<CubeType>::type;
    for (size_t i = 0; i < input.n_slices; i++)
    {
      output.slice(i).submat(
          linspace<UVec>(0, output.n_rows - 1, input.n_rows),
          linspace<UVec>(0, output.n_cols - 1, input.n_cols)) = input.slice(i);
    }
  }

  /**
   * Insert zeros between the units of the given input data.
   * Note: This function should be used before using padding layer.
   *
   * @param input The input to be padded.
   * @param output The padded output data.
   */
  void InsertZeros(const MatType& input, MatType& output)
  {
    const size_t expandedRows = strideWidth *
        (this->inputDimensions[0] - 1) + 1;
    const size_t expandedCols = strideHeight *
        (this->inputDimensions[1] - 1) + 1;
    const size_t outputSize = expandedRows * expandedCols * inMaps
        * higherInDimensions;

    if (output.size() != outputSize)
      output = zeros(outputSize, batchSize);

    CubeType reshapedInput, reshapedOutput;
    MakeAlias(reshapedInput, input, this->inputDimensions[0],
        this->inputDimensions[1], inMaps * higherInDimensions * batchSize);
    MakeAlias(reshapedOutput, output, expandedRows,
        expandedCols, inMaps * higherInDimensions * batchSize);
    InsertZeros(reshapedInput, reshapedOutput);
  }

  //! Locally-stored number of output channels.
  size_t maps;

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

  //! Locally-stored output padding width.
  size_t outputPadW;

  //! Locally-stored output padding height.
  size_t outputPadH;

  //! Locally-stored useBias.
  bool useBias;

  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored weight object.
  CubeType weight;

  //! Locally-stored bias term object.
  MatType bias;

  //! Locally-stored expandInput term.
  bool expandInput;

  //! Locally-stored padInput term.
  bool padInput;

  //! Locally-stored transformed input parameter.
  CubeType inputTemp;

  //! Locally-stored padding layer.
  Padding<MatType> padding;

  //! Locally-stored padding layer for backward pass.
  Padding<MatType> paddingBackward;

  //! Type of padding.
  std::string paddingType;

  //! Locally-cached number of input maps.
  size_t inMaps;

  //! Locally-cached higher-order input dimensions.
  size_t higherInDimensions;
}; // class TransposedConvolution

} // namespace mlpack

// Include implementation.
#include "transposed_convolution_impl.hpp"

#endif
