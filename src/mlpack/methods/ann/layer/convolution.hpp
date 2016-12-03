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

#include <mlpack/core.hpp>

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
<<<<<<< HEAD
  Convolution();
=======
  Convolution()
  {
    /* Nothing to do here. */
  }
>>>>>>> Refactor ann layer.

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
<<<<<<< HEAD
              const size_t inputHeight = 0);
=======
              const size_t inputHeight = 0) :
      inSize(inSize),
      outSize(outSize),
      kW(kW),
      kH(kH),
      dW(dW),
      dH(dH),
      padW(padW),
      padH(padH),
      inputWidth(inputWidth),
      inputHeight(inputHeight),
      outputWidth(0),
      outputHeight(0)
  {
    weights.set_size((outSize * inSize * kW * kH) + outSize, 1);
  }
>>>>>>> Refactor ann layer.

  /*
   * Set the weight and bias term.
   */
<<<<<<< HEAD
  void Reset();
=======
  void Reset()
  {
    weight = arma::cube(weights.memptr(), kW, kH,
        outSize * inSize, false,false);
    bias = arma::mat(weights.memptr() + weight.n_elem,
        outSize, 1, false, false);
  }
>>>>>>> Refactor ann layer.

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);
=======
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    inputTemp = arma::cube(input.memptr(), inputWidth, inputHeight, inSize);

    if (padW != 0 || padH != 0)
    {
      Pad(inputTemp, padW, padH, inputPaddedTemp);
    }

    size_t wConv = ConvOutSize(inputWidth, kW, dW, padW);
    size_t hConv = ConvOutSize(inputHeight, kH, dH, padH);

    outputTemp = arma::zeros<arma::Cube<eT> >(wConv, hConv, outSize);

    for (size_t outMap = 0, outMapIdx = 0; outMap < outSize; outMap++)
    {
      for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
      {
        arma::Mat<eT> convOutput;

        if (padW != 0 || padH != 0)
        {
          ForwardConvolutionRule::Convolution(inputPaddedTemp.slice(inMap),
              weight.slice(outMapIdx), convOutput, dW, dH);
        }
        else
        {
          ForwardConvolutionRule::Convolution(inputTemp.slice(inMap),
              weight.slice(outMapIdx), convOutput, dW, dH);
        }

        outputTemp.slice(outMap) += convOutput;
      }

      outputTemp.slice(outMap) += bias(outMap);
    }

    output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem, 1);

    outputWidth = outputTemp.n_rows;
    outputHeight = outputTemp.n_cols;
  }
>>>>>>> Refactor ann layer.

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
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    arma::cube mappedError = arma::cube(gy.memptr(),
        outputWidth, outputHeight, outSize);
    gTemp = arma::zeros<arma::Cube<eT> >(inputTemp.n_rows,
        inputTemp.n_cols, inputTemp.n_slices);

    for (size_t outMap = 0, outMapIdx = 0; outMap < outSize; outMap++)
    {
      for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
      {
        arma::Mat<eT> rotatedFilter;
        Rotate180(weight.slice(outMapIdx), rotatedFilter);

        arma::Mat<eT> output;
        BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
            rotatedFilter, output, dW, dH);

        if (padW != 0 || padH != 0)
        {
          gTemp.slice(inMap) += output.submat(rotatedFilter.n_rows / 2,
              rotatedFilter.n_cols / 2,
              rotatedFilter.n_rows / 2 + gTemp.n_rows - 1,
              rotatedFilter.n_cols / 2 + gTemp.n_cols - 1);
        }
        else
        {
          gTemp.slice(inMap) += output;
        }


      }
    }

    g = arma::mat(gTemp.memptr(), gTemp.n_elem, 1);
  }
>>>>>>> Refactor ann layer.

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
<<<<<<< HEAD
                arma::Mat<eT>&& gradient);
=======
                arma::Mat<eT>&& gradient)
  {
    arma::cube mappedError;
    if (padW != 0 && padH != 0)
    {
      mappedError = arma::cube(error.memptr(), outputWidth / padW,
          outputHeight / padH, outSize);
    }
    else
    {
      mappedError = arma::cube(error.memptr(), outputWidth,
          outputHeight, outSize);
    }

    gradientTemp = arma::zeros<arma::Cube<eT> >(weight.n_rows, weight.n_cols,
        weight.n_slices);

    for (size_t outMap = 0, outMapIdx = 0; outMap < outSize; outMap++)
    {
      for (size_t inMap = 0, s = outMap; inMap < inSize; inMap++, outMapIdx++,
          s += outSize)
      {
        arma::Cube<eT> inputSlices;
        if (padW != 0 || padH != 0)
        {
          inputSlices = inputPaddedTemp.slices(inMap, inMap);
        }
        else
        {
          inputSlices = inputTemp.slices(inMap, inMap);
        }

        arma::Cube<eT> deltaSlices = mappedError.slices(outMap, outMap);

        arma::Cube<eT> output;
        GradientConvolutionRule::Convolution(inputSlices, deltaSlices,
            output, dW, dH);

        if ((padW != 0 || padH != 0) &&
            (gradientTemp.n_rows < output.n_rows &&
            gradientTemp.n_cols < output.n_cols))
        {
          for (size_t i = 0; i < output.n_slices; i++)
          {
            arma::mat subOutput = output.slice(i);

            gradientTemp.slice(s) += subOutput.submat(subOutput.n_rows / 2,
                subOutput.n_cols / 2,
                subOutput.n_rows / 2 + gradientTemp.n_rows - 1,
                subOutput.n_cols / 2 + gradientTemp.n_cols - 1);
          }
        }
        else
        {
          for (size_t i = 0; i < output.n_slices; i++)
          {
            gradientTemp.slice(s) += output.slice(i);
          }
        }
      }

      gradient.submat(weight.n_elem + outMap, 0,
          weight.n_elem + outMap, 0) = arma::accu(mappedError.slices(
          outMap, outMap));
    }

    gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(gradientTemp);
  }
>>>>>>> Refactor ann layer.

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
<<<<<<< HEAD
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(inSize, "inSize");
    ar & data::CreateNVP(outSize, "outSize");
    ar & data::CreateNVP(kW, "kW");
    ar & data::CreateNVP(kH, "kH");
    ar & data::CreateNVP(dW, "dW");
    ar & data::CreateNVP(dH, "dH");
    ar & data::CreateNVP(padW, "padW");
    ar & data::CreateNVP(padH, "padH");
    ar & data::CreateNVP(weights, "weights");
    ar & data::CreateNVP(inputWidth, "inputWidth");
    ar & data::CreateNVP(inputHeight, "inputHeight");
    ar & data::CreateNVP(outputWidth, "outputWidth");
    ar & data::CreateNVP(outputHeight, "outputHeight");
  }
>>>>>>> Refactor ann layer.

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

<<<<<<< HEAD
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "convolution_impl.hpp"

=======

} // namespace ann
} // namespace mlpack

>>>>>>> Refactor ann layer.
#endif