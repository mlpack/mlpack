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
    const std::string& paddingTypeIn) :
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
  this->paddingType = util::ToLower(paddingTypeIn);
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
>::ConvolutionType(const ConvolutionType& other) :
    Layer<InputType, OutputType>(other),
    maps(other.maps),
    kernelWidth(other.kernelWidth),
    kernelHeight(other.kernelHeight),
    strideWidth(other.strideWidth),
    strideHeight(other.strideHeight),
    padWLeft(other.padWLeft),
    padWRight(other.padWRight),
    padHBottom(other.padHBottom),
    padHTop(other.padHTop),
    padding(other.padding),
    paddingType(other.paddingType),
    inMaps(other.inMaps),
    higherInDimensions(other.higherInDimensions)
{
  // Nothing to do.
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
>::ConvolutionType(ConvolutionType&& other) :
    Layer<InputType, OutputType>(std::move(other)),
    maps(std::move(other.maps)),
    kernelWidth(std::move(other.kernelWidth)),
    kernelHeight(std::move(other.kernelHeight)),
    strideWidth(std::move(other.strideWidth)),
    strideHeight(std::move(other.strideHeight)),
    padWLeft(std::move(other.padWLeft)),
    padWRight(std::move(other.padWRight)),
    padHBottom(std::move(other.padHBottom)),
    padHTop(std::move(other.padHTop)),
    padding(std::move(other.padding)),
    paddingType(std::move(other.paddingType)),
    inMaps(std::move(other.inMaps)),
    higherInDimensions(std::move(other.higherInDimensions))
{
  // Nothing to do.
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
>&
ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::operator=(const ConvolutionType& other)
{
  if (&other != this)
  {
    Layer<InputType, OutputType>::operator=(other);
    maps = other.maps;
    kernelWidth = other.kernelWidth;
    kernelHeight = other.kernelHeight;
    strideWidth = other.strideWidth;
    strideHeight = other.strideHeight;
    padWLeft = other.padWLeft;
    padWRight = other.padWRight;
    padHBottom = other.padHBottom;
    padHTop = other.padHTop;
    padding = other.padding;
    paddingType = other.paddingType;
    inMaps = other.inMaps;
    higherInDimensions = other.higherInDimensions;
  }

  return *this;
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
>&
ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputType,
    OutputType
>::operator=(ConvolutionType&& other)
{
  if (&other != this)
  {
    Layer<InputType, OutputType>::operator=(std::move(other));
    maps = std::move(other.maps);
    kernelWidth = std::move(other.kernelWidth);
    kernelHeight = std::move(other.kernelHeight);
    strideWidth = std::move(other.strideWidth);
    strideHeight = std::move(other.strideHeight);
    padWLeft = std::move(other.padWLeft);
    padWRight = std::move(other.padWRight);
    padHBottom = std::move(other.padHBottom);
    padHTop = std::move(other.padHTop);
    padding = std::move(other.padding);
    paddingType = std::move(other.paddingType);
    inMaps = std::move(other.inMaps);
    higherInDimensions = std::move(other.higherInDimensions);
  }

  return *this;
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
  MakeAlias(weight, weightPtr, kernelWidth, kernelHeight, maps * inMaps);
  MakeAlias(bias, weightPtr + weight.n_elem, maps, 1);
  MakeAlias(weights, weightPtr, weight.n_elem + bias.n_elem, 1);
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

  // First, perform any padding if necessary.
  const bool usingPadding =
      (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0);
  const size_t paddedRows = this->inputDimensions[0] + padWLeft + padWRight;
  const size_t paddedCols = this->inputDimensions[1] + padHTop + padHBottom;
  if (usingPadding)
  {
    // TODO: double-check with a padding test?
    inputPadded.set_size(paddedRows * paddedCols * inMaps * higherInDimensions,
        input.n_cols);
    padding.Forward(input, inputPadded);
  }

  arma::Cube<typename InputType::elem_type> inputTemp;
  MakeAlias(inputTemp,
      const_cast<InputType&>(usingPadding ? inputPadded : input).memptr(),
      paddedRows, paddedCols, inMaps * higherInDimensions * batchSize);

  MakeAlias(outputTemp, output.memptr(), this->outputDimensions[0],
      this->outputDimensions[1], maps * higherInDimensions * batchSize);
  outputTemp.zeros();

  // We "ignore" dimensions higher than the third---that means that we just pass
  // them through and treat them like different input points.
  //
  // If we eventually have a way to do convolutions for a single kernel
  // in-batch, then this strategy may not be the most efficient solution.
  for (size_t offset = 0; offset < (higherInDimensions * batchSize); ++offset)
  {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;

    // Iterate over output maps.
    for (size_t outMap = 0; outMap < maps; ++outMap)
    {
      // Iterate over input maps (we will apply the filter and sum).
      for (size_t inMap = 0; inMap < inMaps; ++inMap)
      {
        OutputType convOutput;

        ForwardConvolutionRule::Convolution(
            inputTemp.slice(inMap + fullInputOffset),
            weight.slice(outMap),
            convOutput,
            strideWidth,
            strideHeight);

        outputTemp.slice(outMap + fullOutputOffset) += convOutput;
      }

      // Make sure to add the bias.
      outputTemp.slice(outMap + fullOutputOffset) += bias(outMap);
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
>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  arma::Cube<typename OutputType::elem_type> mappedError;
  MakeAlias(mappedError, ((OutputType&) gy).memptr(), this->outputDimensions[0],
      this->outputDimensions[1], higherInDimensions * maps * batchSize);

  MakeAlias(gTemp, g.memptr(), this->inputDimensions[0],
      this->inputDimensions[1], inMaps * higherInDimensions * batchSize);
  gTemp.zeros();

  const bool usingPadding =
      (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0);

  // To perform the backward pass, we need to rotate all the filters.
  arma::Cube<typename OutputType::elem_type> rotatedFilters(weight.n_cols,
      weight.n_rows, weight.n_slices);
  for (size_t map = 0; map < maps; ++map)
  {
    Rotate180(weight.slice(map), rotatedFilters.slice(map));
  }

  // See Forward() for the overall iteration strategy.
  for (size_t offset = 0; offset < (higherInDimensions * batchSize); ++offset)
  {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;

    // Iterate over input maps.
    for (size_t inMap = 0; inMap < inMaps; ++inMap)
    {
      // Iterate over output maps.
      for (size_t outMap = 0; outMap < maps; ++outMap)
      {
        OutputType output;

        BackwardConvolutionRule::Convolution(
            mappedError.slice(outMap + fullOutputOffset),
            rotatedFilters.slice(outMap),
            output,
            1,
            1,
            strideWidth,
            strideHeight);

        if (usingPadding)
        {
          gTemp.slice(inMap + fullInputOffset) += output.submat(padWLeft,
              padHTop, padWLeft + gTemp.n_rows - 1, padHTop + gTemp.n_cols - 1);
        }
        else
        {
          gTemp.slice(inMap + fullInputOffset) += output;
        }
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
  arma::Cube<typename OutputType::elem_type> mappedError;
  MakeAlias(mappedError, ((OutputType&) error).memptr(),
      this->outputDimensions[0], this->outputDimensions[1],
      higherInDimensions * maps * batchSize);

  // We are depending here on `inputPadded` being properly set from a call to
  // Forward().
  const bool usingPadding =
      (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0);
  const size_t paddedRows = this->inputDimensions[0] + padWLeft + padWRight;
  const size_t paddedCols = this->inputDimensions[1] + padHTop + padHBottom;

  arma::Cube<typename InputType::elem_type> inputTemp(
      const_cast<InputType&>(usingPadding ? inputPadded : input).memptr(),
      paddedRows, paddedCols, inMaps * batchSize, false, false);

  // We will make an alias for the gradient, but note that this is only for the
  // convolution map weights!  The bias will be handled by direct accesses into
  // `gradient`.
  gradient.zeros();
  MakeAlias(gradientTemp, gradient.memptr(), weight.n_rows, weight.n_cols,
      weight.n_slices);

  // See Forward() for our iteration strategy.
  for (size_t offset = 0; offset < higherInDimensions * batchSize; ++offset)
  {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;

    for (size_t outMap = 0; outMap < maps; ++outMap)
    {
      for (size_t inMap = 0; inMap < inMaps; ++inMap)
      {
        OutputType output;
        GradientConvolutionRule::Convolution(
            inputTemp.slice(inMap + fullInputOffset),
            mappedError.slice(outMap + fullOutputOffset),
            output,
            strideWidth,
            strideHeight);

        // TODO: understand this conditional.  Is it needed?
        if (gradientTemp.n_rows < output.n_rows ||
            gradientTemp.n_cols < output.n_cols)
        {
          gradientTemp.slice(outMap) += output.submat(0, 0,
              gradientTemp.n_rows - 1, gradientTemp.n_cols - 1);
        }
        else if (gradientTemp.n_rows > output.n_rows ||
                 gradientTemp.n_cols > output.n_cols)
        {
          gradientTemp.slice(outMap).submat(0, 0, output.n_rows - 1,
              output.n_cols - 1) += output;
        }
        else
        {
          gradientTemp.slice(outMap) += output;
        }
      }

      gradient[weight.n_elem + outMap] += arma::accu(mappedError.slice(outMap +
          fullOutputOffset));
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
  ar(CEREAL_NVP(inMaps));
  ar(CEREAL_NVP(higherInDimensions));
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
