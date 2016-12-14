/**
 * @file convolution_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONVOLUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONVOLUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "convolution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Convolution()
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
Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Convolution(
    const size_t inSize,
    const size_t outSize,
    const size_t kW,
    const size_t kH,
    const size_t dW,
    const size_t dH,
    const size_t padW,
    const size_t padH,
    const size_t inputWidth,
    const size_t inputHeight) :
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

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
void Convolution<
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
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
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

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
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

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void Convolution<
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

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename Archive>
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Serialize(
    Archive& ar, const unsigned int /* version */)
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

} // namespace ann
} // namespace mlpack

#endif