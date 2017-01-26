/**
 * @file convolution.hpp
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

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Convolution class. The Convolution class represents a
 * single layer of a neural network.
 *
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class Convolution
{
public:
  //! Create the Convolution object.
  Convolution();

  /**
   * Create the Convolution object using the specified number of input maps,
   * output maps, filter size, stride and padding parameter.
   *
   * @param inSize The number of input maps.
   * @param outSize The number of output maps.
   * @param kW Width of the filter/kernel.
   * @param kH Height of the filter/kernel.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param inputWidth The widht of the input data.
   * @param inputHeight The height of the input data.
   */
  Convolution(const size_t inSize,
              const size_t outSize,
              const size_t kW,
              const size_t kH,
              const size_t dW = 1,
              const size_t dH = 1,
              const size_t padW = 0,
              const size_t padH = 0,
              const size_t inputWidth = 0,
              const size_t inputHeight = 0);

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
  template<typename eT>
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

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

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:

  /*
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param p The size of the padding (width or height).
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t p)
  {
    return std::floor(size + p * 2 - k) / s + 1;
  }

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

  /*
   * Pad the given input data.
   *
   * @param input The input to be padded.
   * @param wPad Padding width of the input.
   * @param hPad Padding height of the input.
   * @param output The padded output data.
   */
  template<typename eT>
  void Pad(const arma::Mat<eT>& input,
           size_t wPad,
           size_t hPad,
           arma::Mat<eT>& output)
  {
    if (output.n_rows != input.n_rows + wPad * 2 ||
        output.n_cols != input.n_cols + hPad * 2)
    {
      output = arma::zeros(input.n_rows + wPad * 2, input.n_cols + hPad * 2);
    }

    output.submat(wPad, hPad, wPad + input.n_rows - 1,
        hPad + input.n_cols - 1) = input;
  }

  /*
   * Pad the given input data.
   *
   * @param input The input to be padded.
   * @param wPad Padding width of the input.
   * @param hPad Padding height of the input.
   * @param output The padded output data.
   */
  template<typename eT>
  void Pad(const arma::Cube<eT>& input,
           size_t wPad,
           size_t hPad,
           arma::Cube<eT>& output)
  {
    output = arma::zeros(input.n_rows + wPad * 2,
        input.n_cols + hPad * 2, input.n_slices);

    for (size_t i = 0; i < input.n_slices; ++i)
    {
      Pad<double>(input.slice(i), wPad, hPad, output.slice(i));
    }
  }

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored filter/kernel width.
  size_t kW;

  //! Locally-stored filter/kernel height.
  size_t kH;

  //! Locally-stored stride of the filter in x-direction.
  size_t dW;

  //! Locally-stored stride of the filter in y-direction.
  size_t dH;

  //! Locally-stored padding width.
  size_t padW;

  //! Locally-stored padding height.
  size_t padH;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored weight object.
  arma::cube weight;

  //! Locally-stored bias term object.
  arma::mat bias;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored transformed output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed padded input parameter.
  arma::cube inputPaddedTemp;

  //! Locally-stored transformed error parameter.
  arma::cube gTemp;

  //! Locally-stored transformed gradient parameter.
  arma::cube gradientTemp;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Convolution

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "convolution_impl.hpp"

#endif