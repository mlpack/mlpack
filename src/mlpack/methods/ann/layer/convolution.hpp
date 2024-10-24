/**
 * @file methods/ann/layer/convolution.hpp
 * @author Marcus Edel
 *
 * Definition of the Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_LAYER_CONVOLUTION_HPP

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
 * Implementation of the Convolution class. The Convolution class represents a
 * single layer of a neural network.
 * Example usage:
 *
 * Suppose we want to pass a matrix M (2744x100) to a `Convolution` layer;
 * in this example, `M` was obtained from "flattening" 100 images (or Mel
 * cepstral coefficients, if we talk about speech, or whatever you like) of
 * dimension 196x14. In other words, the first 196 columns of each row of M
 * will be made of the 196 columns of the first row of each of the 100 images
 * (or Mel cepstral coefficients). Then the next 295 columns of M (196 - 393)
 * will be made of the 196 columns of the second row of the 100 images (or Mel
 * cepstral coefficients), etc.  Given that the size of our 2-D input images is
 * 196x14, the parameters for our `Convolution` layer will be something like
 * this:
 *
 * ```
 * Convolution<> c(1, // Number of input activation maps.
 *                 14, // Number of output activation maps.
 *                 3, // Filter width.
 *                 3, // Filter height.
 *                 1, // Stride along width.
 *                 1, // Stride along height.
 *                 0, // Padding width.
 *                 0, // Padding height.
 *                 196, // Input width.
 *                 14); // Input height.
 * ```
 *
 * This `Convolution<>` layer will treat each column of the input matrix `M` as
 * a 2-D image (or object) of the original 196x14 size, using this as the input
 * for the 14 filters of this example.
 *
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template <
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename MatType = arma::mat
>
class ConvolutionType : public Layer<MatType>
{
 public:
  using CubeType = typename GetCubeType<MatType>::type;

  //! Create the ConvolutionType object.
  ConvolutionType();

  /**
   * Create the ConvolutionType object using the specified number of output
   * maps, filter size, stride and padding parameter.
   *
   * @param maps The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param paddingType The type of padding ("valid" or "same"). Defaults to
   *    "none".  If not specified or "none", the values for `padW` and `padH`
   *    will be used.
   * @param useBias Whether or not to use a bias with the convolution.
   */
  ConvolutionType(const size_t maps,
                  const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const size_t padW = 0,
                  const size_t padH = 0,
                  const std::string& paddingType = "none",
                  const bool useBias = true);

  /**
   * Create the Convolution object using the specified number of input maps,
   * output maps, filter size, stride and padding parameter.
   *
   * @param maps The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW A two-value tuple indicating padding widths of the input.  The
   *      first value is the padding for the left side; the second value is the
   *      padding on the right side.
   * @param padH A two-value tuple indicating padding heights of the input.  The
   *      first value is the padding for the top; the second value is the
   *      padding on the bottom.
   * @param paddingType The type of padding ("valid" or "same"). Defaults to
   *      "none".  If not specified or "none", the values for `padW` and `padH`
   *      will be used.
   * @param useBias Whether or not to use a bias with the convolution.
   */
  ConvolutionType(const size_t maps,
                  const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t strideWidth,
                  const size_t strideHeight,
                  const std::tuple<size_t, size_t>& padW,
                  const std::tuple<size_t, size_t>& padH,
                  const std::string& paddingType = "none",
                  const bool useBias = true);

  //! Clone the ConvolutionType object. This handles polymorphism correctly.
  ConvolutionType* Clone() const { return new ConvolutionType(*this); }

  //! Copy the given ConvolutionType (but not weights).
  ConvolutionType(const ConvolutionType& layer);

  //! Take ownership of the given ConvolutionType (but not weights).
  ConvolutionType(ConvolutionType&&);

  //! Copy the given ConvolutionType (but not weights).
  ConvolutionType& operator=(const ConvolutionType& layer);

  //! Take ownership of the given ConvolutionType (but not weights).
  ConvolutionType& operator=(ConvolutionType&& layer);

  // Virtual destructor.
  virtual ~ConvolutionType() { }

  /*
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
   * @param output The propagated data (f(x)) resulting from Forward()
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
  CubeType const& Weight() const
  {
    return weight;
  }
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
    return (maps * inMaps * kernelWidth * kernelHeight) +
        (useBias ? maps : 0);
  }

  //! Compute the output dimensions of the layer based on `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param pSideOne The size of the padding (width or height) on one side.
   * @param pSideTwo The size of the padding (width or height) on another side.
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t pSideOne,
                     const size_t pSideTwo)
  {
    return std::floor(size + pSideOne + pSideTwo - k) / s + 1;
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
    output = CubeType(input.n_rows, input.n_cols, input.n_slices);

    // * left-right flip, up-down flip */
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
    // * left-right flip, up-down flip */
    output = fliplr(flipud(input));
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

  //! Locally-stored useBias.
  bool useBias;

  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored weight object.
  CubeType weight;

  //! Locally-stored bias term object.
  MatType bias;

  //! Locally-stored transformed output parameter.
  CubeType outputTemp;

  //! Locally-stored transformed padded input parameter.
  MatType inputPadded;

  //! Locally-stored transformed error parameter.
  CubeType gTemp;

  //! Locally-stored transformed gradient parameter.
  CubeType gradientTemp;

  //! Locally-stored padding layer.
  PaddingType<MatType> padding;

  //! Locally-stored padding layer for backward pass.
  PaddingType<MatType> paddingBackward;

  //! Type of padding.
  std::string paddingType;

  //! Locally-cached number of input maps.
  size_t inMaps;
  //! Locally-cached higher-order input dimensions.
  size_t higherInDimensions;

  //! Locally-stored apparent width.
  size_t apparentWidth;

  //! Locally-stored apparent height.
  size_t apparentHeight;
}; // class Convolution

// Standard Convolution layer.
using Convolution = ConvolutionType<NaiveConvolution<ValidConvolution>,
                                    NaiveConvolution<FullConvolution>,
                                    NaiveConvolution<ValidConvolution>,
                                    arma::mat>;

} // namespace mlpack

// Include implementation.
#include "convolution_impl.hpp"

#endif
