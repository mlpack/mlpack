/**
 * @file methods/ann/layer/pixel_shuffle_impl.hpp
 * @author Anjishnu Mukherjee
 * @author Abhinav Anand
 *
 * Implementation of the PixelShuffle class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_IMPL_HPP

// In case it hasn't yet been included.
#include "pixel_shuffle.hpp"

namespace mlpack {

template<typename InputDataType, typename OutputDataType>
PixelShuffle<InputDataType, OutputDataType>::PixelShuffle() :
    PixelShuffle(0, 0, 0, 0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
PixelShuffle<InputDataType, OutputDataType>::PixelShuffle(
    const size_t upscaleFactor,
    const size_t height,
    const size_t width,
    const size_t size) :
    upscaleFactor(upscaleFactor),
    height(height),
    width(width),
    size(size),
    batchSize(0),
    outputHeight(0),
    outputWidth(0),
    sizeOut(0),
    reset(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void PixelShuffle<InputDataType, OutputDataType>::Forward(
  const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  if (!reset)
  {
    batchSize = input.n_cols;
    sizeOut = size / std::pow(upscaleFactor, 2);
    outputHeight = height * upscaleFactor;
    outputWidth = width * upscaleFactor;
    reset = true;
  }

  output.zeros(outputHeight * outputWidth * sizeOut, batchSize);
  for (size_t n = 0; n < batchSize; n++)
  {
    arma::cube inputTemp(const_cast<arma::mat&>(input).memptr(), height,
        width, size * batchSize, false, false);
    arma::cube outputTemp(const_cast<arma::mat&>(output).memptr(),
        outputHeight, outputWidth, sizeOut * batchSize, false, false);

    for (size_t c = 0; c < sizeOut; c++)
    {
      for (size_t h = 0; h < outputHeight; h++)
      {
        for (size_t w = 0; w < outputWidth; w++)
        {
          size_t height_index = h / upscaleFactor;
          size_t width_index = w / upscaleFactor;
          size_t channel_index = (upscaleFactor * (h % upscaleFactor)) +
              (w % upscaleFactor) + (c * std::pow(upscaleFactor, 2));
          outputTemp(w, h, c + n * sizeOut) = inputTemp(width_index,
              height_index, channel_index + n * size);
        }
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void PixelShuffle<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  g.zeros(arma::size(input));
  for (size_t n = 0; n < batchSize; n++)
  {
    arma::cube gyTemp(const_cast<arma::mat&>(gy).memptr(), outputHeight,
        outputWidth, sizeOut * batchSize, false, false);
    arma::cube gTemp(const_cast<arma::mat&>(g).memptr(), height, width,
        size * batchSize, false, false);

    for (size_t c = 0; c < sizeOut; c++)
    {
      for (size_t h = 0; h < outputHeight; h++)
      {
        for (size_t w = 0; w < outputWidth; w++)
        {
          size_t height_index = h / upscaleFactor;
          size_t width_index = w / upscaleFactor;
          size_t channel_index = (upscaleFactor * (h % upscaleFactor)) +
              (w % upscaleFactor) + (c * std::pow(upscaleFactor, 2));
          gTemp(width_index, height_index, channel_index + n * size) =
              gyTemp(w, h, c + n * sizeOut);
        }
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PixelShuffle<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(delta));
  ar(CEREAL_NVP(outputParameter));
  ar(CEREAL_NVP(upscaleFactor));
  ar(CEREAL_NVP(height));
  ar(CEREAL_NVP(width));
  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(outputHeight));
  ar(CEREAL_NVP(outputHeight));
  ar(CEREAL_NVP(outputWidth));
  ar(CEREAL_NVP(sizeOut));
}

} // namespace mlpack

#endif
