/**
 * @file transposed_convolution_impl.hpp
 * @author Shikhar Jaiswal
 * @author Marcus Edel
 *
 * Implementation of the Transposed Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_TRANSPOSED_CONVOLUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_TRANSPOSED_CONVOLUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "transposed_convolution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
TransposedConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::TransposedConvolution()
{
  // Nothing to do here.
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
TransposedConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::TransposedConvolution(
    const size_t inSize,
    const size_t outSize,
    const size_t kW,
    const size_t kH,
    const size_t dW,
    const size_t dH,
    const size_t padW,
    const size_t padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t outputWidth,
    const size_t outputHeight) :
    inSize(inSize),
    outSize(outSize),
    kW(kW),
    kH(kH),
    dW(dW),
    dH(dH),
    padW(kW - padW - 1),
    padH(kH - padH - 1),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(outputWidth),
    outputHeight(outputHeight)
{
  weights.set_size((outSize * inSize * kW * kH) + outSize, 1);
  // TODO: Use the Padding layer.
  // padding = new Padding<>(this->padW, this->padW, this->padH, this->padH);

  aW = (outputWidth + kW - 2 * this->padW - 2) % dW;
  aH = (outputHeight + kH - 2 * this->padH - 2) % dH;

  // Check if the output height and width are possible given the other
  // parameters of the layer.
  if (outputWidth != dW * (inputWidth - 1) + aW + 2 * this->padW + 2 - kW ||
      outputHeight != dH * (inputHeight - 1) + aH + 2 * this->padH + 2 - kH)
  {
    Log::Fatal << "The output width / output height is not possible given "
        << "the other parameters of the layer." << std::endl;
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
void TransposedConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Reset()
{
    weight = arma::cube(weights.memptr(), kW, kH,
        outSize * inSize, false, false);
    bias = arma::mat(weights.memptr() + weight.n_elem,
        outSize, 1, false, false);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void TransposedConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  batchSize = input.n_cols;
  inputTemp = arma::cube(const_cast<arma::Mat<eT>&&>(input).memptr(),
      inputWidth, inputHeight, inSize * batchSize, false, false);

  if (dW > 1 || dH > 1)
  {
    InsertZeros(inputTemp, dW, dH, inputExpandedTemp);

    if (padW != 0 || padH != 0 || aW != 0 || aH != 0)
      Pad(inputExpandedTemp, padW, padH, aW, aH, inputPaddedTemp);
    else
    {
      inputPaddedTemp = arma::Cube<eT>(inputExpandedTemp.memptr(),
          inputExpandedTemp.n_rows, inputExpandedTemp.n_cols,
          inputExpandedTemp.n_slices, false, false);;
    }
  }
  else if (padW != 0 || padH != 0 || aW != 0 || aH != 0)
  {
    Pad(inputTemp, padW, padH, aW, aH, inputPaddedTemp);
  }

  output.set_size(outputWidth * outputHeight * outSize, batchSize);
  outputTemp = arma::Cube<eT>(output.memptr(), outputWidth, outputHeight,
      outSize * batchSize, false, false);
  outputTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      outSize * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % outSize == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> convOutput, rotatedFilter;
      Rotate180(weight.slice(outMapIdx), rotatedFilter);

      if (dW > 1 || dH > 1 || padW != 0 || padH != 0 || aW != 0 || aH != 0)
      {
        ForwardConvolutionRule::Convolution(inputPaddedTemp.slice(inMap +
            batchCount * inSize), rotatedFilter, convOutput, 1, 1);
      }
      else
      {
        ForwardConvolutionRule::Convolution(inputTemp.slice(inMap +
            batchCount * inSize), rotatedFilter, convOutput, 1, 1);
      }

      outputTemp.slice(outMap) += convOutput;
    }

    outputTemp.slice(outMap) += bias(outMap % outSize);
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void TransposedConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  arma::Cube<eT> mappedError(gy.memptr(), outputWidth, outputHeight,
      outSize * batchSize, false, false);

  arma::Cube<eT> mappedErrorPadded;
  if ((int)(kW - padW - 1) > 0 || (int)(kH - padH - 1) > 0)
    Pad(mappedError, kW - padW - 1, kH - padH - 1, 0, 0, mappedErrorPadded);

  g.set_size(inputTemp.n_rows * inputTemp.n_cols * inSize, batchSize);
  gTemp = arma::Cube<eT>(g.memptr(), inputTemp.n_rows,
      inputTemp.n_cols, inputTemp.n_slices, false, false);

  gTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      outSize * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % outSize == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> output;

      if ((int)(kW - padW - 1) > 0 || (int)(kH - padH - 1) > 0)
      {
        BackwardConvolutionRule::Convolution(mappedErrorPadded.slice(outMap),
            weight.slice(outMapIdx), output, dW, dH);
      }
      else
      {
        BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
            weight.slice(outMapIdx), output, dW, dH);
      }

      gTemp.slice(inMap + batchCount * inSize) += output;
    }
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void TransposedConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Gradient(
    const arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  arma::Cube<eT> mappedError(error.memptr(), outputWidth,
      outputHeight, outSize * batchSize, false, false);

  gradient.set_size(weights.n_elem, 1);
  gradientTemp = arma::Cube<eT>(gradient.memptr(), weight.n_rows,
      weight.n_cols, weight.n_slices, false, false);
  gradientTemp.zeros();

  arma::Mat<eT> inputSlice, output, deltaSlice, rotatedOutput;

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      outSize * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % outSize == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    deltaSlice = mappedError.slice(outMap);

    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      if (dW > 1 || dH > 1 || padW != 0 || padH != 0 || aW != 0 || aH != 0)
      {
        inputSlice = inputPaddedTemp.slice(inMap + batchCount * inSize);
      }
      else
      {
        inputSlice = inputTemp.slice(inMap + batchCount * inSize);
      }

      GradientConvolutionRule::Convolution(inputSlice, deltaSlice,
          output, 1, 1);
      Rotate180(output, rotatedOutput);
      gradientTemp.slice(outMapIdx) += rotatedOutput;
    }

    gradient.submat(weight.n_elem + (outMap % outSize), 0, weight.n_elem +
        (outMap % outSize), 0) = arma::accu(mappedError.slices(outMap, outMap));
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename Archive>
void TransposedConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(kW);
  ar & BOOST_SERIALIZATION_NVP(kH);
  ar & BOOST_SERIALIZATION_NVP(dW);
  ar & BOOST_SERIALIZATION_NVP(dH);
  ar & BOOST_SERIALIZATION_NVP(padW);
  ar & BOOST_SERIALIZATION_NVP(padH);
  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);

  if (Archive::is_loading::value)
  {
    weights.set_size((outSize * inSize * kW * kH) + outSize, 1);

    aW = (outputWidth + kW - 2 * padW - 2) % dW;
    aH = (outputHeight + kH - 2 * padH - 2) % dH;
  }
}

} // namespace ann
} // namespace mlpack

#endif
