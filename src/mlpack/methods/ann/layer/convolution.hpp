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

#include "layer_types.hpp"
#include "padding.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class ConvolutionType : public Layer<InputType, OutputType>
{
 public:
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
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  ConvolutionType(const size_t maps,
                  const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const size_t padW = 0,
                  const size_t padH = 0,
                  const std::string& paddingType = "None");

  /**
   * Create the Convolution object using the specified number of input maps,
   * output maps, filter size, stride and padding parameter.
   *
   * @param maps The number of output maps.
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
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  ConvolutionType(const size_t maps,
                  const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t strideWidth,
                  const size_t strideHeight,
                  const std::tuple<size_t, size_t>& padW,
                  const std::tuple<size_t, size_t>& padH,
                  const std::string& paddingType = "None");

  //! Clone the ConvolutionType object. This handles polymorphism correctly.
  ConvolutionType* Clone() const { return new ConvolutionType(*this); }

  //! Copy constructor.
  Convolution(const Convolution& layer);

  //! Move constructor.
  Convolution(Convolution&&);

  //! Copy assignment operator.
  Convolution& operator=(const Convolution& layer);

  //! Move assignment operator.
  Convolution& operator=(Convolution&& layer);

  /*
   * Set the weight and bias term.
   */
  void SetWeights(typename OutputType::elem_type* weightsPtr);

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

  //! Get the weight of the layer as a cube.
  arma::cube const& Weight() const { return weight; }
  //! Modify the weight of the layer as a cube.
  arma::cube& Weight() { return weight; }

  //! Get the bias of the layer.
  OutputType const& Bias() const { return bias; }
  //! Modify the bias of the layer.
  OutputType& Bias() { return bias; }

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
    return (maps * inMaps * higherInDimensions * kernelWidth * kernelHeight) +
        maps;
  }

  void ComputeOutputDimensions()
  {
    // First, we must make sure the padding sizes are up to date, which we can
    // now do since inputDimensions is set correctly.
    if (paddingType == "valid")
    {
      padWLeft = 0;
      padWRight = 0;
      padHTop = 0;
      padHBottom = 0;
    }
    else if (paddingType == "same")
    {
      InitializeSamePadding();
    }

    padding = ann::Padding(padWLeft, padWRight, padHTop, padHBottom);
    padding.InputDimensions() = this->inputDimensions;
    padding.ComputeOutputDimensions();

    // We must ensure that the output has at least 3 dimensions, since we will
    // be adding some number of maps to the output.
    this->outputDimensions = std::vector<size_t>(
        std::max(this->inputDimensions.size(), size_t(3)), 1);
    this->outputDimensions[0] = ConvOutSize(this->inputDimensions[0],
        kernelWidth, strideWidth, padWLeft, padWRight);
    this->outputDimensions[1] = ConvOutSize(this->inputDimensions[1],
        kernelHeight, strideHeight, padHTop, padHBottom);

    inMaps = (this->inputDimensions.size() >= 3) ? this->inputDimensions[2] : 1;

    // Compute and cache the total number of input maps.
    higherInDimensions = 1;
    for (size_t i = 3; i < this->inputDimensions.size(); ++i)
    {
      higherInDimensions *= this->inputDimensions[i];
      this->outputDimensions[i] = this->inputDimensions[i];
    }

    this->outputDimensions[2] = maps;
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
  template<typename eT>
  void Rotate180(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = arma::Cube<eT>(input.n_rows, input.n_cols, input.n_slices);

    // * left-right flip, up-down flip */
    for (size_t s = 0; s < output.n_slices; s++)
      output.slice(s) = arma::fliplr(arma::flipud(input.slice(s)));
  }

  /**
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

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored weight object.
  arma::Cube<typename OutputType::elem_type> weight;

  //! Locally-stored bias term object.
  OutputType bias;

  //! Locally-stored transformed output parameter.
  arma::Cube<typename OutputType::elem_type> outputTemp;

  //! Locally-stored transformed padded input parameter.
  InputType inputPadded;

  //! Locally-stored transformed error parameter.
  arma::Cube<typename OutputType::elem_type> gTemp;

  //! Locally-stored transformed gradient parameter.
  arma::Cube<typename OutputType::elem_type> gradientTemp;

  //! Locally-stored padding layer.
  ann::Padding padding;

  //! Type of padding.
  std::string paddingType;

  //! Locally-cached number of input maps.
  size_t inMaps;
  //! Locally-cached higher-order input dimensions.
  size_t higherInDimensions;
}; // class Convolution

// Standard Convolution layer.
typedef ConvolutionType<
    NaiveConvolution<ValidConvolution>,
    NaiveConvolution<FullConvolution>,
    NaiveConvolution<ValidConvolution>,
    arma::mat,
    arma::mat
> Convolution;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "convolution_impl.hpp"

#endif
