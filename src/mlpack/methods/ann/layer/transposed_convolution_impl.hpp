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
    const size_t padW,
    const size_t padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t outputWidth,
    const size_t outputHeight,
    const std::string& paddingType) :
    TransposedConvolution(
      inSize,
      outSize,
      kernelWidth,
      kernelHeight,
      strideWidth,
      strideHeight,
      std::tuple<size_t, size_t>(padW, padW),
      std::tuple<size_t, size_t>(padH, padH),
      inputWidth,
      inputHeight,
      outputWidth,
      outputHeight,
      paddingType)
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
    const std::tuple<size_t, size_t>& padW,
    const std::tuple<size_t, size_t>& padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t outputWidth,
    const size_t outputHeight,
    const std::string& paddingType) :
    inSize(inSize),
    outSize(outSize),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWLeft(std::get<0>(padW)),
    padWRight(std::get<1>(padW)),
    padHBottom(std::get<1>(padH)),
    padHTop(std::get<0>(padH)),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(outputWidth),
    outputHeight(outputHeight)
{
  weights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
      1);
  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  if (paddingTypeLow == "valid")
  {
    // Set Padding to 0.
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  const size_t totalPadWidth = padWLeft + padWRight;
  const size_t totalPadHeight = padHTop + padHBottom;

  aW = (outputWidth + totalPadWidth - kernelWidth) % strideWidth;
  aH = (outputHeight + totalPadHeight - kernelHeight) % strideHeight;

  const size_t padWidthLeftForward = kernelWidth - padWLeft - 1;
  const size_t padHeightTopForward = kernelHeight - padHTop - 1;
  const size_t padWidthRightForward = kernelWidth - padWRight - 1;
  const size_t padHeightBottomtForward = kernelHeight - padHBottom - 1;

  paddingForward = ann::Padding<>(padWidthLeftForward,
      padWidthRightForward + aW, padHeightTopForward,
      padHeightBottomtForward + aH);
  paddingBackward = ann::Padding<>(padWLeft, padWRight, padHTop, padHBottom);

  // Check if the output height and width are possible given the other
  // parameters of the layer.
  if (outputWidth != strideWidth * (inputWidth - 1) +
      aW + kernelWidth - totalPadWidth ||
      outputHeight != strideHeight * (inputHeight - 1) +
      aH + kernelHeight - totalPadHeight)
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

    if (paddingForward.PadWLeft() != 0 || paddingForward.PadWRight() != 0 ||
        paddingForward.PadHTop() != 0 || paddingForward.PadHBottom() != 0)
    {
      inputPaddedTemp.set_size(inputExpandedTemp.n_rows +
          paddingForward.PadWLeft() + paddingForward.PadWRight(),
          inputExpandedTemp.n_cols + paddingForward.PadHTop() +
          paddingForward.PadHBottom(), inputExpandedTemp.n_slices);

      for (size_t i = 0; i < inputExpandedTemp.n_slices; ++i)
      {
        paddingForward.Forward(std::move(inputExpandedTemp.slice(i)),
            std::move(inputPaddedTemp.slice(i)));
      }
    }
    else
    {
      inputPaddedTemp = arma::Cube<eT>(inputExpandedTemp.memptr(),
          inputExpandedTemp.n_rows, inputExpandedTemp.n_cols,
          inputExpandedTemp.n_slices, false, false);;
    }
  }
  else if (paddingForward.PadWLeft() != 0 ||
           paddingForward.PadWRight() != 0 ||
           paddingForward.PadHTop() != 0 ||
           paddingForward.PadHBottom() != 0)
  {
    inputPaddedTemp.set_size(inputTemp.n_rows + paddingForward.PadWLeft() +
        paddingForward.PadWRight(), inputTemp.n_cols +
        paddingForward.PadHTop() + paddingForward.PadHBottom(),
        inputTemp.n_slices);

    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      paddingForward.Forward(std::move(inputTemp.slice(i)),
          std::move(inputPaddedTemp.slice(i)));
    }
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

      if (strideWidth > 1 ||
          strideHeight > 1 ||
          paddingForward.PadWLeft() != 0 ||
          paddingForward.PadWRight() != 0 ||
          paddingForward.PadHTop() != 0 ||
          paddingForward.PadHBottom() != 0)
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
  if (paddingBackward.PadWLeft() != 0 || paddingBackward.PadWRight() != 0 ||
      paddingBackward.PadHTop() != 0 || paddingBackward.PadHBottom() != 0)
  {
    mappedErrorPadded.set_size(mappedError.n_rows +
        paddingBackward.PadWLeft() + paddingBackward.PadWRight(),
        mappedError.n_cols + paddingBackward.PadHTop() +
        paddingBackward.PadHBottom(), mappedError.n_slices);

    for (size_t i = 0; i < mappedError.n_slices; ++i)
    {
      paddingBackward.Forward(std::move(mappedError.slice(i)),
          std::move(mappedErrorPadded.slice(i)));
    }
  }
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

      if (paddingBackward.PadWLeft() != 0 || paddingBackward.PadWRight() != 0 ||
          paddingBackward.PadHTop() != 0 || paddingBackward.PadHBottom() != 0)
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
      if (strideWidth > 1 ||
          strideHeight > 1 ||
          paddingForward.PadWLeft() != 0 ||
          paddingForward.PadWRight() != 0 ||
          paddingForward.PadHTop() != 0 ||
          paddingForward.PadHBottom() != 0)
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
    Archive& ar, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(kernelWidth);
  ar & BOOST_SERIALIZATION_NVP(kernelHeight);
  ar & BOOST_SERIALIZATION_NVP(strideWidth);
  ar & BOOST_SERIALIZATION_NVP(strideHeight);
  if (version == 0)
  {
    // These are now stored in paddingForward and paddingBackward.
    size_t padWidth, padHeight;
    ar & BOOST_SERIALIZATION_NVP(padWidth);
    ar & BOOST_SERIALIZATION_NVP(padHeight);
  }
  ar &BOOST_SERIALIZATION_NVP(padWLeft);
  ar &BOOST_SERIALIZATION_NVP(padWRight);
  ar &BOOST_SERIALIZATION_NVP(padHBottom);
  ar &BOOST_SERIALIZATION_NVP(padHTop);
  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);

  if (version > 0)
  {
    ar & BOOST_SERIALIZATION_NVP(paddingForward);
    ar & BOOST_SERIALIZATION_NVP(paddingBackward);
  }

  if (Archive::is_loading::value)
  {
    weights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
        1);
    size_t totalPadWidth = padWLeft + padWRight;
    size_t totalPadHeight = padHTop + padHBottom;
    aW = (outputWidth + kernelWidth - totalPadWidth - 2) % strideWidth;
    aH = (outputHeight + kernelHeight - totalPadHeight - 2) % strideHeight;
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
>::InitializeSamePadding(){
  /**
   * Using O=s*(I-1) + K -2P + A
   * where
   * s=stride 
   * I=Input Shape
   * K=Kernel Size
   * P=Padding
   */
  const size_t totalHorizontalPadding  = (strideWidth - 1) * inputWidth +
      kernelWidth - strideWidth;
  const size_t totalVerticalPadding = (strideHeight - 1) * inputHeight +
      kernelHeight - strideHeight;

  padWLeft = totalVerticalPadding / 2;
  padWRight = totalVerticalPadding - totalVerticalPadding / 2;
  padHTop = totalHorizontalPadding / 2;
  padHBottom = totalHorizontalPadding - totalHorizontalPadding / 2;

  // If Padding is negative throw a fatal error.
  if (totalHorizontalPadding < 0 || totalVerticalPadding < 0)
  {
    Log::Fatal << "The output width / output height is not possible given "
               << "same padding for the layer." << std::endl;
  }
}

} // namespace ann
} // namespace mlpack

#endif
