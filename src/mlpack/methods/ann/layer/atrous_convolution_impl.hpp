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
  batchSize = input.n_cols;
  inputTemp = arma::cube(const_cast<arma::Mat<eT>&&>(input).memptr(),
      inputWidth, inputHeight, inSize * batchSize, false, false);

  if (padW != 0 || padH != 0)
  {
    Pad(inputTemp, padW, padH, inputPaddedTemp);
  }

  size_t wConv = ConvOutSize(inputWidth, kW, dW, padW, dilationW);
  size_t hConv = ConvOutSize(inputHeight, kH, dH, padH, dilationH);

  output.set_size(wConv * hConv * outSize, batchSize);
  outputTemp = arma::Cube<eT>(output.memptr(), wConv, hConv,
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
      arma::Mat<eT> convOutput;

      if (padW != 0 || padH != 0)
      {
        ForwardConvolutionRule::Convolution(inputPaddedTemp.slice(inMap +
            batchCount * inSize), weight.slice(outMapIdx), convOutput, dW, dH,
            dilationW, dilationH);
      }
      else
      {
        ForwardConvolutionRule::Convolution(inputTemp.slice(inMap +
            batchCount * inSize), weight.slice(outMapIdx), convOutput, dW, dH,
            dilationW, dilationH);
      }

      outputTemp.slice(outMap) += convOutput;
    }

    outputTemp.slice(outMap) += bias(outMap % outSize);
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
  arma::cube mappedError(gy.memptr(), outputWidth, outputHeight,
      outSize * batchSize, false, false);

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
      arma::Mat<eT> output, rotatedFilter;
      Rotate180(weight.slice(outMapIdx), rotatedFilter);

      BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
          rotatedFilter, output, dW, dH, dilationW, dilationH);

      if (padW != 0 || padH != 0)
      {
        gTemp.slice(inMap + batchCount * inSize) += output.submat(padW, padH,
            padW + gTemp.n_rows - 1,
            padH + gTemp.n_cols - 1);
      }
      else
      {
        gTemp.slice(inMap + batchCount * inSize) += output;
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
  arma::cube mappedError(error.memptr(), outputWidth, outputHeight,
      outSize * batchSize, false, false);

  gradient.set_size(weights.n_elem, 1);
  gradientTemp = arma::Cube<eT>(gradient.memptr(), weight.n_rows,
      weight.n_cols, weight.n_slices, false, false);
  gradientTemp.zeros();

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
      arma::Mat<eT> inputSlice;
      if (padW != 0 || padH != 0)
      {
        inputSlice = inputPaddedTemp.slice(inMap + batchCount * inSize);
      }
      else
      {
        inputSlice = inputTemp.slice(inMap + batchCount * inSize);
      }

      arma::Mat<eT> deltaSlice = mappedError.slice(outMap);

      arma::Mat<eT> output;
      GradientConvolutionRule::Convolution(inputSlice, deltaSlice,
          output, dW, dH, 1, 1);

      if (dilationH > 1)
      {
        for (size_t i = 1; i < output.n_cols; i++){
          output.shed_cols(i, i + dilationH - 2);
        }
      }
      if (dilationW > 1)
      {
        for (size_t i = 1; i < output.n_rows; i++){
          output.shed_rows(i, i + dilationW - 2);
        }
      }

      if (gradientTemp.n_rows < output.n_rows ||
          gradientTemp.n_cols < output.n_cols)
      {
        gradientTemp.slice(outMapIdx) += output.submat(0, 0,
            gradientTemp.n_rows - 1, gradientTemp.n_cols - 1);
      }
      else if (gradientTemp.n_rows > output.n_rows ||
          gradientTemp.n_cols > output.n_cols)
      {
        gradientTemp.slice(outMapIdx).submat(0, 0, output.n_rows - 1,
            output.n_cols - 1) += output;
      }
      else
      {
        gradientTemp.slice(outMapIdx) += output;
      }
    }

    gradient.submat(weight.n_elem + (outMap % outSize), 0, weight.n_elem +
        (outMap % outSize), 0) = arma::accu(mappedError.slice(outMap));
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
  ar & BOOST_SERIALIZATION_NVP(dilationW);
  ar & BOOST_SERIALIZATION_NVP(dilationH);

  if (Archive::is_loading::value)
    weights.set_size((outSize * inSize * kW * kH) + outSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif
