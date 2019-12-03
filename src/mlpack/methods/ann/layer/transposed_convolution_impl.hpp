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
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const size_t padWidth,
    const size_t padHeight,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t outputWidth,
    const size_t outputHeight) :
    inSize(inSize),
    outSize(outSize),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWidth(kernelWidth - padWidth - 1),
    padHeight(kernelHeight - padHeight - 1),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(outputWidth),
    outputHeight(outputHeight)
{
  weights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize, 1);
  // TODO: Use the Padding layer.
  // padding = new Padding<>(this->padWidth, this->padWidth, this->padHeight, this->padHeight);

  aW = (outputWidth + kernelWidth - 2 * this->padWidth - 2) % strideWidth;
  aH = (outputHeight + kernelHeight - 2 * this->padHeight - 2) % strideHeight;

  // Check if the output height and width are possible given the other
  // parameters of the layer.
  if (outputWidth != strideWidth * (inputWidth - 1) + aW + 2 * this->padWidth + 2 - kernelWidth ||
      outputHeight != strideHeight * (inputHeight - 1) + aH + 2 * this->padHeight + 2 - kernelHeight)
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
    weight = arma::cube(weights.memptr(), kernelWidth, kernelHeight,
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

  if (strideWidth > 1 || strideHeight > 1)
  {
    InsertZeros(inputTemp, strideWidth, strideHeight, inputExpandedTemp);

    if (padWidth != 0 || padHeight != 0 || aW != 0 || aH != 0)
      Pad(inputExpandedTemp, padWidth, padHeight, aW, aH, inputPaddedTemp);
    else
    {
      inputPaddedTemp = arma::Cube<eT>(inputExpandedTemp.memptr(),
          inputExpandedTemp.n_rows, inputExpandedTemp.n_cols,
          inputExpandedTemp.n_slices, false, false);;
    }
  }
  else if (padWidth != 0 || padHeight != 0 || aW != 0 || aH != 0)
  {
    Pad(inputTemp, padWidth, padHeight, aW, aH, inputPaddedTemp);
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

      if (strideWidth > 1 || strideHeight > 1 || padWidth != 0 || padHeight != 0 || aW != 0 || aH != 0)
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
  if ((int)(kernelWidth - padWidth - 1) > 0 || (int)(kernelHeight - padHeight - 1) > 0)
    Pad(mappedError, kernelWidth - padWidth - 1, kernelHeight - padHeight - 1, 0, 0, mappedErrorPadded);

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

      if ((int)(kernelWidth - padWidth - 1) > 0 || (int)(kernelHeight - padHeight - 1) > 0)
      {
        BackwardConvolutionRule::Convolution(mappedErrorPadded.slice(outMap),
            weight.slice(outMapIdx), output, strideWidth, strideHeight);
      }
      else
      {
        BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
            weight.slice(outMapIdx), output, strideWidth, strideHeight);
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
      if (strideWidth > 1 || strideHeight > 1 || padWidth != 0 || padHeight != 0 || aW != 0 || aH != 0)
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
  ar & BOOST_SERIALIZATION_NVP(kernelWidth);
  ar & BOOST_SERIALIZATION_NVP(kernelHeight);
  ar & BOOST_SERIALIZATION_NVP(strideWidth);
  ar & BOOST_SERIALIZATION_NVP(strideHeight);
  ar & BOOST_SERIALIZATION_NVP(padWidth);
  ar & BOOST_SERIALIZATION_NVP(padHeight);
  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);

  if (Archive::is_loading::value)
  {
    weights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
        1);

    aW = (outputWidth + kernelWidth - 2 * padWidth - 2) % strideWidth;
    aH = (outputHeight + kernelHeight - 2 * padHeight - 2) % strideHeight;
  }
}

} // namespace ann
} // namespace mlpack

#endif
