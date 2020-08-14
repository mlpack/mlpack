/**
 * @file methods/ann/layer/pixel_shuffle_impl.hpp
 * @author Anjishnu Mukherjee
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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
PixelShuffle<InputDataType, OutputDataType>::PixelShuffle() :
    upscaleFactor(0),
    height(0),
    width(0),
    size(0),
    reset(false)
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
    arma::mat inputImage = input.col(n);
    arma::mat outputImage = output.col(n);
    arma::cube inputTemp(const_cast<arma::mat&>(inputImage).memptr(), height,
        width, size, false, false);
    arma::cube outputTemp(const_cast<arma::mat&>(outputImage).memptr(),
        outputHeight, outputWidth, sizeOut, false, false);

    for (size_t c = 0; c < sizeOut ; c++)
    {
      for (size_t h = 0; h < outputHeight; h++)
      {
        for (size_t w = 0; w < outputWidth; w++)
        {
          size_t height_index = h / upscaleFactor;
          size_t width_index = w / upscaleFactor;
          size_t channel_index = (upscaleFactor * (h % upscaleFactor)) +
              (w % upscaleFactor) + (c * std::pow(upscaleFactor, 2));
          outputTemp(w, h, c) = inputTemp(width_index, height_index,
              channel_index);
        }
      }
    }
    output.col(n) = outputImage;
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
    arma::mat gyImage = gy.col(n);
    arma::mat gImage = g.col(n);
    arma::cube gyTemp(const_cast<arma::mat&>(gyImage).memptr(), outputHeight,
        outputWidth, sizeOut, false, false);
    arma::cube gTemp(const_cast<arma::mat&>(gImage).memptr(), height, width,
        size, false, false);

    for (size_t c = 0; c < sizeOut ; c++)
    {
      for (size_t h = 0; h < outputHeight; h++)
      {
        for (size_t w = 0; w < outputWidth; w++)
        {
          size_t height_index = h / upscaleFactor;
          size_t width_index = w / upscaleFactor;
          size_t channel_index = (upscaleFactor * (h % upscaleFactor)) +
              (w % upscaleFactor) + (c * std::pow(upscaleFactor, 2));
          gTemp(width_index, height_index, channel_index) = gyTemp(w, h, c);
        }
      }
    }

    g.col(n) = gImage;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PixelShuffle<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(delta);
  ar & BOOST_SERIALIZATION_NVP(outputParameter);
  ar & BOOST_SERIALIZATION_NVP(upscaleFactor);
  ar & BOOST_SERIALIZATION_NVP(height);
  ar & BOOST_SERIALIZATION_NVP(width);
  ar & BOOST_SERIALIZATION_NVP(size);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(sizeOut);
}

} // namespace ann
} // namespace mlpack

#endif
