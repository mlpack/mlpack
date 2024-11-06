/**
 * @file methods/ann/layer/transposed_convolution_impl.hpp
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

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::TransposedConvolutionType()
{
  // Nothing to do here.
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::TransposedConvolutionType(
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
    TransposedConvolutionType(
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
    typename InputType,
    typename OutputType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::TransposedConvolutionType(
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
  weights.set_size(WeightSize(), 1);
  // Transform paddingType to lowercase.
  const std::string paddingTypeLow = util::ToLower(paddingType);

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

  paddingForward = Padding(padWidthLeftForward,
      padWidthRightForward + aW, padHeightTopForward,
      padHeightBottomtForward + aH);
  paddingBackward = Padding(padWLeft, padWRight, padHTop, padHBottom);

  // Check if the output height and width are possible given the other
  // parameters of the layer.
  if (outputWidth != 0 && outputHeight != 0 &&
      (outputWidth != strideWidth * (inputWidth - 1) +
      aW + kernelWidth - totalPadWidth ||
      outputHeight != strideHeight * (inputHeight - 1) +
      aH + kernelHeight - totalPadHeight))
  {
    Log::Fatal << "The output width / output height is not possible given "
               << "the other parameters of the layer." << std::endl;
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::SetWeights(typename OutputType::elem_type* weightPtr)
{
  weight = arma::Cube<typename OutputType::elem_type>(weightPtr,
      kernelWidth, kernelHeight, outSize * inSize, false, false);
  bias = arma::Mat<typename OutputType::elem_type>(weightsPtr +
      weight.n_elem, outSize, 1, false, false);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::Forward(const InputType& input, OutputType& output)
{
  batchSize = input.n_cols;
  arma::Cube<typename InputType::elem_type> inputTemp(
      const_cast<InputType&>(input).memptr(), inputWidth, inputHeight,
      inSize * batchSize, false, false);

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
        paddingForward.Forward(inputExpandedTemp.slice(i),
            inputPaddedTemp.slice(i));
      }
    }
    else
    {
      inputPaddedTemp = arma::Cube<typename InputType::elem_type>(
          inputExpandedTemp.memptr(), inputExpandedTemp.n_rows,
          inputExpandedTemp.n_cols, inputExpandedTemp.n_slices, false, false);
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
      paddingForward.Forward(inputTemp.slice(i), inputPaddedTemp.slice(i));
    }
  }

  output.set_size(outputWidth * outputHeight * outSize, batchSize);
  outputTemp = arma::Cube<typename OutputType::elem_type>(output.memptr(),
      outputWidth, outputHeight, outSize * batchSize, false, false);
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
      OutputType convOutput, rotatedFilter;
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
    typename InputType,
    typename OutputType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  arma::Cube<typename OutputType::elem_type> mappedError(
      ((OutputType&) gy).memptr(), outputWidth,
      outputHeight, outSize * batchSize, false, false);
  arma::Cube<typename OutputType::elem_type> mappedErrorPadded;
  if (paddingBackward.PadWLeft() != 0 || paddingBackward.PadWRight() != 0 ||
      paddingBackward.PadHTop() != 0 || paddingBackward.PadHBottom() != 0)
  {
    mappedErrorPadded.set_size(mappedError.n_rows +
        paddingBackward.PadWLeft() + paddingBackward.PadWRight(),
        mappedError.n_cols + paddingBackward.PadHTop() +
        paddingBackward.PadHBottom(), mappedError.n_slices);

    for (size_t i = 0; i < mappedError.n_slices; ++i)
    {
      paddingBackward.Forward(mappedError.slice(i),
          mappedErrorPadded.slice(i));
    }
  }
  g.set_size(inputWidth * inputHeight * inSize, batchSize);
  gTemp = arma::Cube<typename OutputType::elem_type>(g.memptr(), inputWidth,
      inputHeight, inSize * batchSize, false, false);

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
      OutputType output;

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
    typename InputType,
    typename OutputType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  arma::Cube<typename OutputType::elem_type> mappedError(
      ((OutputType&) error).memptr(), outputWidth, outputHeight,
      outSize * batchSize, false, false);
  arma::Cube<typename InputType::elem_type> inputTemp(
      const_cast<InputType&>(input).memptr(), inputWidth, inputHeight,
      inSize * batchSize, false, false);

  gradient.set_size(weights.n_elem, 1);
  gradientTemp = arma::Cube<typename InputType::elem_type>(gradient.memptr(),
      weight.n_rows, weight.n_cols, weight.n_slices, false, false);
  gradientTemp.zeros();

  OutputType inputSlice, output, deltaSlice, rotatedOutput;

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
        (outMap % outSize), 0) = accu(mappedError.slices(outMap, outMap));
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
template<typename Archive>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHBottom));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(inputWidth));
  ar(CEREAL_NVP(inputHeight));
  ar(CEREAL_NVP(outputWidth));
  ar(CEREAL_NVP(outputHeight));
  ar(CEREAL_NVP(paddingForward));
  ar(CEREAL_NVP(paddingBackward));

  if (cereal::is_loading<Archive>())
  {
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
    typename InputType,
    typename OutputType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
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
}

} // namespace mlpack

#endif
