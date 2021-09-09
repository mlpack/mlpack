/**
 * @file methods/ann/layer/convolution_impl.hpp
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
    typename InputType,
    typename OutputType
>
ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::ConvolutionType()
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
ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::ConvolutionType(
    const size_t maps,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const size_t padW,
    const size_t padH,
    const std::string& paddingType) :
    ConvolutionType(
      maps,
      kernelWidth,
      kernelHeight,
      strideWidth,
      strideHeight,
      std::tuple<size_t, size_t>(padW, padW),
      std::tuple<size_t, size_t>(padH, padH),
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
ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::ConvolutionType(
    const size_t maps,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const std::tuple<size_t, size_t>& padW,
    const std::tuple<size_t, size_t>& padH,
    const std::string& paddingType) :
    maps(maps),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWLeft(std::get<0>(padW)),
    padWRight(std::get<1>(padW)),
    padHBottom(std::get<1>(padH)),
    padHTop(std::get<0>(padH))
{
  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  util::ToLower(paddingType, paddingTypeLow);

  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  padding = ann::Padding(padWLeft, padWRight, padHTop, padHBottom);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
void ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::SetWeights(typename OutputType::elem_type* weightPtr)
{
  std::cout << "call SetWeights, " << weightPtr << "; kernelWidth " <<
kernelWidth << " kernelHeight " << kernelHeight << " maps " << maps << " totalInMaps " << totalInMaps << "\n";
  weight = arma::Cube<typename OutputType::elem_type>(weightPtr,
      kernelWidth, kernelHeight, maps * totalInMaps, false, false);
  bias = OutputType(weightPtr + weight.n_elem, maps * totalInMaps, 1, false,
      false);
  weights = OutputType(weightPtr, weight.n_elem + bias.n_elem, 1, false, false);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
void ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::Forward(const InputType& input, OutputType& output)
{
  batchSize = input.n_cols;
  arma::Cube<typename InputType::elem_type> inputTemp(
      const_cast<InputType&>(input).memptr(), this->inputDimensions[0],
      this->inputDimensions[1], totalInMaps * batchSize, false, false);

  if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
  {
    inputPaddedTemp.set_size(inputTemp.n_rows + padWLeft + padWRight,
        inputTemp.n_cols + padHTop + padHBottom, inputTemp.n_slices);

    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      padding.Forward(inputTemp.slice(i), inputPaddedTemp.slice(i));
    }
  }

  output.set_size(this->outputDimensions[0] * this->outputDimensions[1] *
      totalInMaps *  maps, batchSize);
  outputTemp = arma::Cube<typename OutputType::elem_type>(output.memptr(),
      this->outputDimensions[0], this->outputDimensions[1],
      maps * totalInMaps * batchSize,
      false, false);
  outputTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      maps * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % maps == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < totalInMaps; inMap++, outMapIdx++)
    {
      OutputType convOutput;

      if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
      {
        ForwardConvolutionRule::Convolution(inputPaddedTemp.slice(inMap +
            batchCount * totalInMaps), weight.slice(outMapIdx), convOutput,
            strideWidth, strideHeight);
      }
      else
      {
        ForwardConvolutionRule::Convolution(inputTemp.slice(inMap +
            batchCount * totalInMaps), weight.slice(outMapIdx), convOutput,
            strideWidth, strideHeight);
      }

      outputTemp.slice(outMap) += convOutput;
    }

    outputTemp.slice(outMap) += bias(outMap % maps);
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
void ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  arma::Cube<typename OutputType::elem_type> mappedError(
      ((OutputType&) gy).memptr(), this->outputDimensions[0],
      this->outputDimensions[1], totalInMaps * maps * batchSize, false, false);

  g.set_size(this->inputDimensions[0] * this->inputDimensions[1] * totalInMaps,
      batchSize);
  gTemp = arma::Cube<typename OutputType::elem_type>(g.memptr(),
      this->inputDimensions[0], this->inputDimensions[1], totalInMaps *
      batchSize, false, false);
  gTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      maps * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % maps == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < totalInMaps; inMap++, outMapIdx++)
    {
      OutputType output, rotatedFilter;
      Rotate180(weight.slice(outMapIdx), rotatedFilter);

      BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
          rotatedFilter, output, strideWidth, strideHeight);

      if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
      {
        gTemp.slice(inMap + batchCount * totalInMaps) += output.submat(padWLeft,
            padHTop, padWLeft + gTemp.n_rows - 1, padHTop + gTemp.n_cols - 1);
      }
      else
      {
        gTemp.slice(inMap + batchCount * totalInMaps) += output;
      }
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
void ConvolutionType<
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
      ((OutputType&) error).memptr(), this->outputDimensions[0],
      this->outputDimensions[1], totalInMaps * maps * batchSize, false, false);
  arma::Cube<typename InputType::elem_type> inputTemp(
      ((InputType&) input).memptr(), this->inputDimensions[0],
      this->inputDimensions[1], totalInMaps * maps * batchSize, false, false);

  gradientTemp = arma::Cube<typename OutputType::elem_type>(gradient.memptr(),
      weight.n_rows, weight.n_cols, weight.n_slices, false, false);
  gradientTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      maps * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % maps == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < totalInMaps; inMap++, outMapIdx++)
    {
      InputType inputSlice;
      if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
      {
        inputSlice = inputPaddedTemp.slice(inMap + batchCount * totalInMaps);
      }
      else
      {
        inputSlice = inputTemp.slice(inMap + batchCount * totalInMaps);
      }

      OutputType deltaSlice = mappedError.slice(outMap);

      OutputType output;
      GradientConvolutionRule::Convolution(inputSlice, deltaSlice, output,
          strideWidth, strideHeight);

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

    gradient.submat(weight.n_elem + (outMap % maps), 0, weight.n_elem +
        (outMap % maps), 0) = arma::accu(mappedError.slice(outMap));
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
void ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::serialize(Archive& ar, const uint32_t /* version*/)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(maps));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHBottom));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(padding));
  ar(CEREAL_NVP(totalInMaps));
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputType,
    typename OutputType
>
void ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::InitializeSamePadding()
{
  /*
   * Using O = (W - F + 2P) / s + 1;
   */
  size_t totalVerticalPadding = (strideWidth - 1) * this->inputDimensions[0] +
      kernelWidth - strideWidth;
  size_t totalHorizontalPadding = (strideHeight - 1) * this->inputDimensions[1]
      + kernelHeight - strideHeight;

  padWLeft = totalVerticalPadding / 2;
  padWRight = totalVerticalPadding - totalVerticalPadding / 2;
  padHTop = totalHorizontalPadding / 2;
  padHBottom = totalHorizontalPadding - totalHorizontalPadding / 2;
}

} // namespace ann
} // namespace mlpack

#endif
