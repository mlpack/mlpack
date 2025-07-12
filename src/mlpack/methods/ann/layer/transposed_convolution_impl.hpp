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

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::TransposedConvolutionType() {
  // Nothing to do here.
}

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::TransposedConvolutionType(
    const size_t maps,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const size_t padW, const size_t padH,
    const std::string &paddingType,
    const bool useBias) :
    TransposedConvolutionType(
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

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::TransposedConvolutionType(
    const size_t maps,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const std::tuple<size_t, size_t>& padW,
    const std::tuple<size_t, size_t>& padH,
    const std::string& paddingType,
    const bool useBias  ) :
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
  this->paddingType = util::ToLower(paddingType);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::TransposedConvolutionType(const TransposedConvolutionType& other) :
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
    higherInDimensions(other.higherInDimensions)
{
  // Nothing to do.
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::TransposedConvolutionType(TransposedConvolutionType&& other) :
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
    higherInDimensions(std::move(other.higherInDimensions))
{
  // Nothing to do.
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>&
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::operator=(const TransposedConvolutionType& other)
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
  }

  return *this;
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>&
TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::operator=(TransposedConvolutionType&& other)
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
  }

  return *this;
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
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
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Forward(const MatType& input, MatType& output)
{
  batchSize = input.n_cols;

  // If there are non-zero padding values, we will need to pad the input.
  // If we are expanding the input, we will need to pad the expanded input.
  // If we are not expanding the input, we will use the original input.
  if (expandInput) {
    InsertZeros(input, inputExpanded);
  }

  if (usingPadding)
  {
    inputPadded.set_size(padding.OutputDimensions()[0]
        * padding.OutputDimensions()[1] * inMaps * higherInDimensions, batchSize);
    padding.Forward(expandInput ? inputExpanded : input, inputPadded);
  }

  MakeAlias(inputTemp,
      usingPadding ? inputPadded : (expandInput ? inputExpanded : input),
      padding.OutputDimensions()[0], padding.OutputDimensions()[1],
      inMaps * higherInDimensions * batchSize);

  MakeAlias(outputTemp, output, this->outputDimensions[0],
      this->outputDimensions[1], maps * higherInDimensions * batchSize);
  outputTemp.zeros();

  // weights need to be flipped for the forward pass
  CubeType rotatedFilters(weight.n_rows, weight.n_cols, weight.n_slices);
  #pragma omp parallel for schedule(static)
  for (size_t map = 0; map < (size_t)(maps * inMaps); ++map)
  {
    Rotate180(weight.slice(map), rotatedFilters.slice(map));
  }

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
    #pragma omp parallel for
    for (size_t outMap = 0; outMap < (size_t)maps; ++outMap)
    {
      MatType &convOutput = outputTemp.slice(outMap + fullOutputOffset);
      // Iterate over input maps (we will apply the filter and sum).
      for (size_t inMap = 0; inMap < inMaps; ++inMap)
      {
        ForwardConvolutionRule::Convolution(
            inputTemp.slice(inMap + fullInputOffset),
            rotatedFilters.slice((outMap * inMaps) + inMap),
            convOutput,
            1,
            1,
            1,
            1,
            true);
      }

      // Make sure to add the bias.
      if (useBias)
        convOutput += bias(outMap);
    }
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Backward(const MatType & /* input */,
            const MatType & /* output */,
            const MatType &gy,
            MatType &g)
{
  MatType errorPadded;

  if (usingPaddingBackward) {
    errorPadded.set_size(paddingBackward.OutputDimensions()[0]
        * paddingBackward.OutputDimensions()[1] * maps * higherInDimensions,
        batchSize);

    paddingBackward.Forward(gy, errorPadded);
  }

  CubeType mappedError;
  MakeAlias(mappedError,
      (usingPaddingBackward ? errorPadded : gy),
      paddingBackward.OutputDimensions()[0],
      paddingBackward.OutputDimensions()[1], 
      maps * higherInDimensions * batchSize);

  MakeAlias(gTemp, g, this->inputDimensions[0], this->inputDimensions[1],
            inMaps * higherInDimensions * batchSize);
  gTemp.zeros();

  // See Forward() for the overall iteration strategy.
  #pragma omp parallel for schedule(dynamic)
  for (size_t offset = 0; offset < (higherInDimensions * batchSize); ++offset) {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;

    // Iterate over input maps.
    for (size_t inMap = 0; inMap < (size_t)inMaps; ++inMap) {
      // Iterate over output maps.
      for (size_t outMap = 0; outMap < maps; ++outMap) {
        BackwardConvolutionRule::Convolution(
            mappedError.slice(outMap + fullOutputOffset),
            weight.slice((outMap * inMaps) + inMap),
            gTemp.slice(inMap + fullInputOffset),
            strideWidth,
            strideHeight,
            1,
            1,
            true);
      }
    }
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Gradient(
    const MatType& /* input */,
    const MatType& error,
    MatType& gradient)
{
  CubeType mappedError;
  MakeAlias(mappedError, error, this->outputDimensions[0],
      this->outputDimensions[1], maps * higherInDimensions * batchSize);

  // We will make an alias for the gradient, but note that this is only for the
  // convolution map weights!  The bias will be handled by direct accesses into
  // `gradient`.
  gradient.zeros();
  MakeAlias(gradientTemp, gradient, kernelWidth, kernelHeight, inMaps * maps);

  MatType curError, rotatedCurError;

  // See Forward() for our iteration strategy.
  for (size_t offset = 0; offset < higherInDimensions * batchSize; ++offset) {
    const size_t fullInputOffset = offset * inMaps;
    const size_t fullOutputOffset = offset * maps;

    #pragma omp parallel for
    for (size_t outMap = 0; outMap < (size_t)maps; ++outMap) {
      for (size_t inMap = 0; inMap < inMaps; ++inMap) {
        GradientConvolutionRule::Convolution(
            inputTemp.slice(inMap + fullInputOffset),
            mappedError.slice(outMap + fullOutputOffset),
            curError,
            1,
            1,
            1,
            1,
            false);
        Rotate180(curError, rotatedCurError);
        gradientTemp.slice((outMap * inMaps) + inMap) += rotatedCurError;
      }

      if (useBias) {
        gradient[weight.n_elem + outMap] =
            accu(mappedError.slice(outMap + fullOutputOffset));
      }
    }
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::ComputeOutputDimensions()
{
  // First, we must make sure the padding sizes are up to date,
  // which we can now do since inputDimensions is set correctly.
  if (paddingType == "valid") {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  } else if (paddingType == "same") {
    InitializeSamePadding();
  }

  // If strideWidth or strideHeight is greater than 1, we will expand the input
  // by inserting zeros between the elements. This is necessary to ensure that
  // the output dimensions are correct for the forward pass.
  expandInput = (strideWidth > 1 || strideHeight > 1);

  // Calculate padding for the forward pass of transposed convolution
  // based on the padding for the forward pass of regular convolution.
  const size_t padWLeftForward = kernelWidth - padWLeft - 1;
  const size_t padWRightForward = kernelWidth - padWRight - 1;
  const size_t padHTopForward = kernelHeight - padHTop - 1;
  const size_t padHBottomForward = kernelHeight - padHBottom - 1;
  padding = PaddingType<MatType>(padWLeftForward, padWRightForward,
      padHTopForward, padHBottomForward);
  
  // Padding is applied after input expansion, so padding layer
  // for the forward pass must account for the expanded size.
  padding.InputDimensions() = this->inputDimensions;
  if (expandInput) {
    padding.InputDimensions()[0] = strideWidth
        * (this->inputDimensions[0] - 1) + 1;
    padding.InputDimensions()[1] = strideWidth 
        * (this->inputDimensions[1] - 1) + 1;
  }
  
  // We only pad the input if there is a non zero padding value.
  padding.ComputeOutputDimensions();
  usingPadding = (padding.PadWLeft() != 0 ||
                  padding.PadWRight() != 0 ||
                  padding.PadHTop() != 0 || 
                  padding.PadHBottom() != 0);

  // TODO: Add this alignment back along with output padding
  // aW = (outputWidth + kernelWidth - totalPadWidth - 2) % strideWidth;
  // aH = (outputHeight + kernelHeight - totalPadHeight - 2) % strideHeight;

  // We must ensure that the output has at least 3 dimensions, since we will
  // be adding some number of maps to the output.
  this->outputDimensions = std::vector<size_t>(std::max(
      this->inputDimensions.size(), size_t(3)), 1);
  this->outputDimensions[0] = TConvOutSize(this->inputDimensions[0],
      kernelWidth, strideWidth, padWLeft, padWRight);
  this->outputDimensions[1] = TConvOutSize(this->inputDimensions[1],
      kernelHeight, strideHeight, padHTop, padHBottom);

  // Compute and cache the total number of input maps.
  inMaps = (this->inputDimensions.size() >= 3) ? this->inputDimensions[2] : 1;

  // dimensions higher than the third are treated as different input points.
  higherInDimensions = 1;
  for (size_t i = 3; i < this->inputDimensions.size(); ++i) {
    higherInDimensions *= this->inputDimensions[i];
    this->outputDimensions[i] = this->inputDimensions[i];
  }

  // Backward transposed conv uses same padding as regular conv.
  // Input dims are the forward output dims of transposed conv.
  paddingBackward = PaddingType<MatType>(padWLeft, padWRight,
      padHTop, padHBottom);
  paddingBackward.InputDimensions() = this->outputDimensions;
  paddingBackward.ComputeOutputDimensions();
  usingPaddingBackward = (paddingBackward.PadWLeft() != 0 ||
                          paddingBackward.PadWRight() != 0 ||
                          paddingBackward.PadHTop() != 0 ||
                          paddingBackward.PadHBottom() != 0);

  this->outputDimensions[2] = maps;
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
template<typename Archive>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
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
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void TransposedConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::InitializeSamePadding()
{
  /**
   * Compute 'same' padding using the standard conv formula:
   * O = (W - F + 2P) / S + 1
   * Transpose conv padding is later derived as: p' = k - p - 1
   * in ComputeOutputDimensions()
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
