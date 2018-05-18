/**
 * @file atrous_convolution_impl.hpp
 * @author Aarush Gupta
 * @author Shikhar Jaiswal
 *
 * Implementation of the Atrous Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "atrous_convolution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::AtrousConvolution()
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
AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::AtrousConvolution(
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
    const size_t dilationW,
    const size_t dilationH) :
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
    outputHeight(0),
    dilationW(dilationW),
    dilationH(dilationH)
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
void AtrousConvolution<
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
void AtrousConvolution<
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

  size_t wConv = ConvOutSize(inputWidth, kW, dW, padW, dilationW);
  size_t hConv = ConvOutSize(inputHeight, kH, dH, padH, dilationH);

  output.set_size(wConv * hConv * outSize, 1);
  outputTemp = arma::Cube<eT>(output.memptr(), wConv, hConv, outSize,
      false, false);
  outputTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0; outMap < outSize; outMap++)
  {
    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> convOutput;
      if (padW != 0 || padH != 0)
      {
        ForwardConvolutionRule::Convolution(inputPaddedTemp.slice(inMap),
            weight.slice(outMapIdx), convOutput, dW, dH, dilationW, dilationH);
      }
      else
      {
        ForwardConvolutionRule::Convolution(inputTemp.slice(inMap),
            weight.slice(outMapIdx), convOutput, dW, dH, dilationW, dilationH);
      }
      outputTemp.slice(outMap) += convOutput;
    }

    outputTemp.slice(outMap) += bias(outMap);
  }

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
void AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  arma::cube mappedError(gy.memptr(), outputWidth, outputHeight, outSize,
      false, false);

  g.set_size(inputTemp.n_rows * inputTemp.n_cols * inputTemp.n_slices, 1);
  gTemp = arma::Cube<eT>(g.memptr(), inputTemp.n_rows,
      inputTemp.n_cols, inputTemp.n_slices, false, false);
  gTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0; outMap < outSize; outMap++)
  {
    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> output, rotatedFilter;
      Rotate180(weight.slice(outMapIdx), rotatedFilter);

      BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
          rotatedFilter, output, dW, dH, dilationW, dilationH);

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
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void AtrousConvolution<
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

  gradient.set_size(weights.n_elem, 1);
  gradientTemp = arma::Cube<eT>(gradient.memptr(), weight.n_rows, weight.n_cols,
      weight.n_slices, false, false);
  gradientTemp.zeros();

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

      arma::Cube<eT> output, reducedOutput;
      GradientConvolutionRule::Convolution(inputSlices, deltaSlices,
          output, dW, dH, 1, 1);
      arma::Mat<eT> reducedMat;
      reducedOutput = arma::zeros<arma::Cube<eT> >(kW, kH, output.n_slices);

      for (size_t j = 0; j < output.n_slices; j++)
      {
        reducedMat = output.slice(j);
        if (dilationH > 1)
        {
          for (size_t i = 1; i < reducedMat.n_cols; i++){
            reducedMat.shed_cols(i, i + dilationH - 2);
          }
        }
        if (dilationW > 1)
        {
          for (size_t i = 1; i < reducedMat.n_rows; i++){
            reducedMat.shed_rows(i, i + dilationW - 2);
          }
        }
        reducedOutput.slice(j) = reducedMat;
      }

      if ((padW != 0 || padH != 0) &&
          (gradientTemp.n_rows < reducedOutput.n_rows &&
          gradientTemp.n_cols < reducedOutput.n_cols))
      {
        for (size_t i = 0; i < reducedOutput.n_slices; i++)
        {
          gradientTemp.slice(s) += reducedOutput.slice(i).submat(
              reducedOutput.n_rows / 2, reducedOutput.n_cols / 2,
              reducedOutput.n_rows / 2 + gradientTemp.n_rows - 1,
              reducedOutput.n_cols / 2 + gradientTemp.n_cols - 1);
        }
      }
      else
      {
        for (size_t i = 0; i < reducedOutput.n_slices; i++)
          gradientTemp.slice(s) += reducedOutput.slice(i);
      }
    }

    gradient.submat(weight.n_elem + outMap, 0, weight.n_elem + outMap, 0) =
        arma::accu(mappedError.slices(outMap, outMap));
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
void AtrousConvolution<
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
  ar & BOOST_SERIALIZATION_NVP(dilationW);
  ar & BOOST_SERIALIZATION_NVP(dilationH);

  if (Archive::is_loading::value)
    weights.set_size((outSize * inSize * kW * kH) + outSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif
