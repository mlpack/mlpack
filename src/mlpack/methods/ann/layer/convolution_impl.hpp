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

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Convolution() : Layer<MatType>()
{
  // Nothing to do here.
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Convolution(
    const size_t maps,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const size_t padW,
    const size_t padH,
    const std::string& paddingType,
    const bool useBias) :
    Convolution(
      maps,
      kernelWidth,
      kernelHeight,
      strideWidth,
      strideHeight,
      std::tuple<size_t, size_t>(padW, padW),
      std::tuple<size_t, size_t>(padH, padH),
      paddingType,
      useBias)
{
  // Nothing to do here.
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Convolution(
    const size_t maps,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const std::tuple<size_t, size_t>& padW,
    const std::tuple<size_t, size_t>& padH,
    const std::string& paddingTypeIn,
    const bool useBias) :
    Layer<MatType>(),
    maps(maps),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWLeft(std::get<0>(padW)),
    padWRight(std::get<1>(padW)),
    padHBottom(std::get<1>(padH)),
    padHTop(std::get<0>(padH)),
    useBias(useBias)
{
  // Transform paddingType to lowercase.
  this->paddingType = util::ToLower(paddingTypeIn);
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Convolution(const Convolution& other) :
    Layer<MatType>(other),
    maps(other.maps),
    kernelWidth(other.kernelWidth),
    kernelHeight(other.kernelHeight),
    strideWidth(other.strideWidth),
    strideHeight(other.strideHeight),
    padWLeft(other.padWLeft),
    padWRight(other.padWRight),
    padHBottom(other.padHBottom),
    padHTop(other.padHTop),
    useBias(other.useBias),
    padding(other.padding),
    paddingBackward(other.paddingBackward),
    paddingType(other.paddingType),
    inMaps(other.inMaps),
    higherInDimensions(other.higherInDimensions),
    apparentWidth(other.apparentWidth),
    apparentHeight(other.apparentHeight)
{
  // Nothing to do.
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Convolution(Convolution&& other) :
    Layer<MatType>(std::move(other)),
    maps(std::move(other.maps)),
    kernelWidth(std::move(other.kernelWidth)),
    kernelHeight(std::move(other.kernelHeight)),
    strideWidth(std::move(other.strideWidth)),
    strideHeight(std::move(other.strideHeight)),
    padWLeft(std::move(other.padWLeft)),
    padWRight(std::move(other.padWRight)),
    padHBottom(std::move(other.padHBottom)),
    padHTop(std::move(other.padHTop)),
    useBias(std::move(other.useBias)),
    padding(std::move(other.padding)),
    paddingBackward(std::move(other.paddingBackward)),
    paddingType(std::move(other.paddingType)),
    inMaps(std::move(other.inMaps)),
    higherInDimensions(std::move(other.higherInDimensions)),
    apparentWidth(std::move(other.apparentWidth)),
    apparentHeight(std::move(other.apparentHeight))
{
  // Nothing to do.
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>&
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::operator=(const Convolution& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    maps = other.maps;
    kernelWidth = other.kernelWidth;
    kernelHeight = other.kernelHeight;
    strideWidth = other.strideWidth;
    strideHeight = other.strideHeight;
    padWLeft = other.padWLeft;
    padWRight = other.padWRight;
    padHBottom = other.padHBottom;
    padHTop = other.padHTop;
    useBias = other.useBias;
    padding = other.padding;
    paddingBackward = other.paddingBackward;
    paddingType = other.paddingType;
    inMaps = other.inMaps;
    higherInDimensions = other.higherInDimensions;
    apparentWidth = other.apparentWidth;
    apparentHeight = other.apparentHeight;
  }

  return *this;
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>&
Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::operator=(Convolution&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    maps = std::move(other.maps);
    kernelWidth = std::move(other.kernelWidth);
    kernelHeight = std::move(other.kernelHeight);
    strideWidth = std::move(other.strideWidth);
    strideHeight = std::move(other.strideHeight);
    padWLeft = std::move(other.padWLeft);
    padWRight = std::move(other.padWRight);
    padHBottom = std::move(other.padHBottom);
    padHTop = std::move(other.padHTop);
    useBias = std::move(other.useBias);
    padding = std::move(other.padding);
    paddingBackward = std::move(other.paddingBackward);
    paddingType = std::move(other.paddingType);
    inMaps = std::move(other.inMaps);
    higherInDimensions = std::move(other.higherInDimensions);
    apparentWidth = std::move(other.apparentWidth);
    apparentHeight = std::move(other.apparentHeight);
  }

  return *this;
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
void Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weight, weightsIn, kernelWidth, kernelHeight, maps * inMaps);
  if (useBias)
  {
    MakeAlias(bias, weightsIn, maps, 1, weight.n_elem);
    MakeAlias(weights, weightsIn, weight.n_elem + bias.n_elem, 1);
  }
  else
  {
    MakeAlias(weights, weightsIn, weight.n_elem, 1);
  }
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
void Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Forward(const MatType& input, MatType& output)
{
  batchSize = input.n_cols;

  // First, perform any padding if necessary.
  const bool usingPadding =
      (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0);
  const size_t paddedRows = this->inputDimensions[0] + padWLeft + padWRight;
  const size_t paddedCols = this->inputDimensions[1] + padHTop + padHBottom;
  if (usingPadding)
  {
    inputPadded.set_size(paddedRows * paddedCols * inMaps * higherInDimensions,
        input.n_cols);
    padding.Forward(input, inputPadded);
  }

  output.zeros();

  // We "ignore" dimensions higher than the third---that means that we just pass
  // them through and treat them like different input points.
  //
  // If we eventually have a way to do convolutions for a single kernel
  // in-batch, then this strategy may not be the most efficient solution.
  #pragma omp parallel for schedule(dynamic) private(outputTemp)
  for (size_t offset = 0; offset < (higherInDimensions * batchSize); ++offset)
  {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;
    CubeType inputTemp;

    MakeAlias(inputTemp, (usingPadding ? inputPadded : input), paddedRows,
        paddedCols, inMaps, fullInputOffset * paddedRows * paddedCols);

    MakeAlias(outputTemp, output, this->outputDimensions[0],
        this->outputDimensions[1], maps, fullOutputOffset *
        this->outputDimensions[0] * this->outputDimensions[1]);

    ForwardConvolutionRule::Convolution(
        inputTemp,
        weight,
        outputTemp,
        strideWidth,
        strideHeight,
        1,
        1,
        true);

    // Make sure to add the bias.
    if (useBias)
    {
      for (size_t outMap = 0; outMap < (size_t) maps; ++outMap)
        outputTemp.slice(outMap) += bias(outMap);
    }
  }
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
void Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  CubeType mappedError;
  MakeAlias(mappedError, gy, this->outputDimensions[0],
      this->outputDimensions[1], higherInDimensions * maps * batchSize);

  MakeAlias(gTemp, g, this->inputDimensions[0], this->inputDimensions[1],
      inMaps * higherInDimensions * batchSize);
  gTemp.zeros();

  const bool usingPadding =
      (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0);

  // To perform the backward pass, we need to rotate all the filters.
  CubeType rotatedFilters(weight.n_rows,
      weight.n_cols, weight.n_slices);

  // To perform the backward pass, we need to dilate all the mappedError.
  CubeType dilatedMappedError;
  if (strideHeight == 1 && strideWidth == 1)
  {
    MakeAlias(dilatedMappedError, mappedError, mappedError.n_rows,
        mappedError.n_cols, mappedError.n_slices);
  }
  else
  {
    dilatedMappedError.zeros(mappedError.n_rows * strideWidth -
        (strideWidth - 1), mappedError.n_cols * strideHeight -
        (strideHeight - 1), mappedError.n_slices);
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t i = 0; i < mappedError.n_slices; ++i)
    {
      for (size_t j = 0; j < mappedError.n_cols; ++j)
      {
        for (size_t k = 0; k < mappedError.n_rows; ++k)
        {
          dilatedMappedError(k * strideWidth, j * strideHeight, i)
              = mappedError(k, j, i);
        }
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t map = 0; map < (size_t) (maps * inMaps); ++map)
  {
    Rotate180(weight.slice(map), rotatedFilters.slice(map));
  }

  MatType output(apparentWidth * apparentHeight * inMaps * higherInDimensions,
      batchSize);

  // See Forward() for the overall iteration strategy.
  #pragma omp parallel for schedule(dynamic) private(outputTemp)
  for (size_t offset = 0; offset < (higherInDimensions * batchSize); ++offset)
  {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;

    CubeType rotatedFiltersTemp;

    MakeAlias(outputTemp, output, apparentWidth, apparentHeight, inMaps,
        fullInputOffset * apparentWidth * apparentHeight);
    // Iterate over output maps.
    for (size_t outMap = 0; outMap < maps; ++outMap)
    {
      MakeAlias(rotatedFiltersTemp, rotatedFilters, rotatedFilters.n_rows,
          rotatedFilters.n_cols, inMaps,
          outMap * inMaps * rotatedFilters.n_rows * rotatedFilters.n_cols);
      BackwardConvolutionRule::Convolution(
          dilatedMappedError.slice(outMap + fullOutputOffset),
          rotatedFiltersTemp,
          outputTemp,
          1,
          1,
          1,
          1,
          true);
    }
  }
  MatType temp(padding.OutputDimensions()[0] * padding.OutputDimensions()[1] *
      inMaps * higherInDimensions, batchSize);
  CubeType tempCube;
  MakeAlias(tempCube, temp, padding.OutputDimensions()[0],
      padding.OutputDimensions()[1], inMaps * higherInDimensions * batchSize);
  paddingBackward.Forward(output, temp);
  if (usingPadding)
  {
    gTemp = tempCube.tube(
        padWLeft,
        padHTop,
        padWLeft + gTemp.n_rows - 1,
        padHTop + gTemp.n_cols - 1);
  }
  else
  {
    gTemp = tempCube;
  }
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
void Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  CubeType mappedError;
  MakeAlias(mappedError, error, this->outputDimensions[0],
      this->outputDimensions[1], higherInDimensions * maps * batchSize);

  // We are depending here on `inputPadded` being properly set from a call to
  // Forward().
  const bool usingPadding =
      (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0);

  MatType temp(apparentWidth * apparentHeight * inMaps * higherInDimensions,
      batchSize);
  paddingBackward.Backward(input, {} /* unused */,
      usingPadding ? inputPadded : input, temp);

  // We will make an alias for the gradient, but note that this is only for the
  // convolution map weights!  The bias will be handled by direct accesses into
  // `gradient`.
  gradient.zeros();
  MakeAlias(gradientTemp, gradient, weight.n_rows, weight.n_cols,
      weight.n_slices);

  MatType tempSlice;

  // See Forward() for our iteration strategy.
  #pragma omp parallel for schedule(dynamic) private(tempSlice)
  for (size_t offset = 0; offset < higherInDimensions * batchSize; ++offset)
  {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;

    CubeType mappedErrorTemp;
    MakeAlias(mappedErrorTemp, error, this->outputDimensions[0],
        this->outputDimensions[1], maps, fullOutputOffset *
        this->outputDimensions[0] * this->outputDimensions[1]);

    for (size_t inMap = 0; inMap < inMaps; ++inMap)
    {
      CubeType gradientTempTemp(gradientTemp.n_rows, gradientTemp.n_cols,
          maps);
      // Make an alias of the slice directly instead of using a cube to avoid
      // the overhead of creating a cube every time this function is called
      MakeAlias(tempSlice, temp, apparentWidth, apparentHeight,
          (inMap + fullInputOffset) * (apparentWidth * apparentHeight));

      GradientConvolutionRule::Convolution(
          tempSlice,
          mappedErrorTemp,
          gradientTempTemp,
          1,
          1,
          strideWidth,
          strideHeight,
          true);

      // Reorder convolution output slices.
      #pragma omp critical
      for (size_t outMap = 0; outMap < (size_t) maps; ++outMap)
      {
        gradientTemp.slice((outMap * inMaps) + inMap) +=
            gradientTempTemp.slice(outMap);
      }
    }

    if (useBias)
    {
      for (size_t outMap = 0; outMap < (size_t) maps; ++outMap)
      {
        #pragma omp atomic update
        gradient[weight.n_elem + outMap] += accu(mappedErrorTemp
            .slice(outMap));
      }
    }
  }
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
void Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::ComputeOutputDimensions()
{
  // First, we must make sure the padding sizes are up to date, which we can
  // now do since inputDimensions is set correctly.
  if (paddingType == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingType == "same")
  {
    InitializeSamePadding();
  }

  padding = Padding<MatType>(padWLeft, padWRight, padHTop, padHBottom);
  padding.InputDimensions() = this->inputDimensions;
  padding.ComputeOutputDimensions();

  // We must ensure that the output has at least 3 dimensions, since we will
  // be adding some number of maps to the output.
  this->outputDimensions = std::vector<size_t>(
      std::max(this->inputDimensions.size(), size_t(3)), 1);
  this->outputDimensions[0] = ConvOutSize(this->inputDimensions[0],
      kernelWidth, strideWidth, padWLeft, padWRight);
  this->outputDimensions[1] = ConvOutSize(this->inputDimensions[1],
      kernelHeight, strideHeight, padHTop, padHBottom);

  inMaps = (this->inputDimensions.size() >= 3) ? this->inputDimensions[2] : 1;

  // Compute and cache the total number of input maps.
  higherInDimensions = 1;
  for (size_t i = 3; i < this->inputDimensions.size(); ++i)
  {
    higherInDimensions *= this->inputDimensions[i];
    this->outputDimensions[i] = this->inputDimensions[i];
  }

  apparentWidth = (this->outputDimensions[0] - 1) * strideWidth + kernelWidth;
  apparentHeight = (this->outputDimensions[1] - 1) * strideHeight +
      kernelHeight;

  paddingBackward = Padding<MatType>(0, padding.OutputDimensions()[0] -
      apparentWidth, 0, padding.OutputDimensions()[1] - apparentHeight);
  paddingBackward.InputDimensions() = std::vector<size_t>({ apparentWidth,
      apparentHeight, inMaps * higherInDimensions });
  paddingBackward.ComputeOutputDimensions();

  this->outputDimensions[2] = maps;
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
template<typename Archive>
void Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::serialize(Archive& ar, const uint32_t /* version*/)
{
  ar(cereal::base_class<Layer<MatType>>(this));

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
  ar(CEREAL_NVP(useBias));
  ar(CEREAL_NVP(padding));
  ar(CEREAL_NVP(paddingType));
  ar(CEREAL_NVP(inMaps));
  ar(CEREAL_NVP(higherInDimensions));
}

template<
    typename MatType,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule
>
void Convolution<
    MatType,
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule
>::InitializeSamePadding()
{
  /**
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

} // namespace mlpack

#endif
