/**
 * @file methods/ann/layer/adaptive_max_pooling_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of the Adaptive Max Pooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "adaptive_max_pooling.hpp"

namespace mlpack {

template<typename MatType>
AdaptiveMaxPoolingType<MatType>::AdaptiveMaxPoolingType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template <typename MatType>
AdaptiveMaxPoolingType<MatType>::AdaptiveMaxPoolingType(
    const size_t outputWidth,
    const size_t outputHeight) :
    Layer<MatType>(),
    poolingLayer(MaxPoolingType<MatType>(1, 1)),
    outputWidth(outputWidth),
    outputHeight(outputHeight)
{
  // Nothing to do here.
}

template<typename MatType>
AdaptiveMaxPoolingType<MatType>::AdaptiveMaxPoolingType(
    const AdaptiveMaxPoolingType& other) :
    Layer<MatType>(other),
    poolingLayer(other.poolingLayer),
    outputWidth(other.outputWidth),
    outputHeight(other.outputHeight)
{
  // Nothing to do here.
}

template<typename MatType>
AdaptiveMaxPoolingType<MatType>::AdaptiveMaxPoolingType(
    AdaptiveMaxPoolingType&& other) :
    Layer<MatType>(std::move(other)),
    poolingLayer(std::move(other.poolingLayer)),
    outputWidth(std::move(other.outputWidth)),
    outputHeight(std::move(other.outputHeight))
{
  // Nothing to do here.
}

template<typename MatType>
AdaptiveMaxPoolingType<MatType>&
AdaptiveMaxPoolingType<MatType>::operator=(const AdaptiveMaxPoolingType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    poolingLayer = other.poolingLayer;
    outputWidth = other.outputWidth;
    outputHeight = other.outputHeight;
  }

  return *this;
}

template<typename MatType>
AdaptiveMaxPoolingType<MatType>&
AdaptiveMaxPoolingType<MatType>::operator=(AdaptiveMaxPoolingType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    poolingLayer = std::move(other.poolingLayer);
    outputWidth = std::move(other.outputWidth);
    outputHeight = std::move(other.outputHeight);
  }

  return *this;
}

template<typename MatType>
void AdaptiveMaxPoolingType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  poolingLayer.Training() = this->training;
  poolingLayer.Forward(input, output);
}

template<typename MatType>
void AdaptiveMaxPoolingType<MatType>::Backward(
  const MatType& input,
  const MatType& output,
  const MatType& gy,
  MatType& g)
{
  poolingLayer.Backward(input, output, gy, g);
}

template<typename MatType>
void AdaptiveMaxPoolingType<MatType>::ComputeOutputDimensions()
{
  // The AdaptiveMaxPooling layer only affects the first two dimensions.
  this->outputDimensions = this->inputDimensions;

  this->outputDimensions[0] = outputWidth;
  this->outputDimensions[1] = outputHeight;

  if (outputWidth > this->inputDimensions[0] ||
        outputHeight > this->inputDimensions[1])
  {
    Log::Fatal << "Given output shape (" << outputWidth << ", "
      << outputHeight << ") is not possible for given input shape ("
      << this->inputDimensions[0] << ", " << this->inputDimensions[1]
      << ")." << std::endl;
  }
  poolingLayer.InputDimensions() = this->inputDimensions;
  poolingLayer.StrideWidth() = std::floor(this->inputDimensions[0] /
      outputWidth);
  poolingLayer.StrideHeight() = std::floor(this->inputDimensions[1] /
      outputHeight);

  poolingLayer.KernelWidth() = this->inputDimensions[0] -
      (outputWidth - 1) * poolingLayer.StrideWidth();
  poolingLayer.KernelHeight() = this->inputDimensions[1] -
      (outputHeight - 1) * poolingLayer.StrideHeight();

  poolingLayer.ComputeOutputDimensions();
}

template<typename MatType>
template<typename Archive>
void AdaptiveMaxPoolingType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(poolingLayer));
  ar(CEREAL_NVP(outputWidth));
  ar(CEREAL_NVP(outputHeight));
}

} // namespace mlpack

#endif
