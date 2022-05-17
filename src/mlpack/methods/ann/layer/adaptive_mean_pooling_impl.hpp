/**
 * @file methods/ann/layer/adaptive_mean_pooling_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of the Adaptive Mean Pooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MEAN_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MEAN_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "adaptive_mean_pooling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename MatType>
AdaptiveMeanPoolingType<MatType>::AdaptiveMeanPoolingType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template <typename MatType>
AdaptiveMeanPoolingType<MatType>::AdaptiveMeanPoolingType(
    const size_t outputWidth,
    const size_t outputHeight) :
    AdaptiveMeanPoolingType(std::tuple<size_t, size_t>(outputWidth, outputHeight))
{
  // Nothing to do here.
}

template <typename MatType>
AdaptiveMeanPoolingType<MatType>::AdaptiveMeanPoolingType(
    const std::tuple<size_t, size_t>& outputShape) : 
    Layer<MatType>(),
    outputWidth(std::get<0>(outputShape)),
    outputHeight(std::get<1>(outputShape))
{
  poolingLayer = ann::MeanPoolingType<MatType>(0, 0);
}

template<typename MatType>
AdaptiveMeanPoolingType<MatType>::AdaptiveMeanPoolingType(
    const AdaptiveMeanPoolingType& other) :
    Layer<MatType>(other),
    poolingLayer(other.poolingLayer),
    outputWidth(other.outputWidth),
    outputHeight(other.outputHeight)
{
  // Nothing to do here.
}

template<typename MatType>
AdaptiveMeanPoolingType<MatType>::AdaptiveMeanPoolingType(
    AdaptiveMeanPoolingType&& other) :
    Layer<MatType>(std::move(other)),
    poolingLayer(std::move(other.poolingLayer)),
    outputWidth(std::move(other.outputWidth)),
    outputHeight(std::move(other.outputHeight))
{
  // Nothing to do here.
}

template<typename MatType>
AdaptiveMeanPoolingType<MatType>&
AdaptiveMeanPoolingType<MatType>::operator=(const AdaptiveMeanPoolingType& other)
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
AdaptiveMeanPoolingType<MatType>&
AdaptiveMeanPoolingType<MatType>::operator=(AdaptiveMeanPoolingType&& other)
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
void AdaptiveMeanPoolingType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  poolingLayer.Training() = this->training;
  poolingLayer.Forward(input, output);
}

template<typename MatType>
void AdaptiveMeanPoolingType<MatType>::Backward(
  const MatType& input,
  const MatType& gy,
  MatType& g)
{
  poolingLayer.Backward(input, gy, g);
}

template<typename MatType>
void AdaptiveMeanPoolingType<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;

  this->outputDimensions[0] = outputWidth;
  this->outputDimensions[1] = outputHeight;

  InitializeAdaptivePadding();

  // Higher dimensions are not modified.
}

template<typename MatType>
template<typename Archive>
void AdaptiveMeanPoolingType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(outputWidth));
  ar(CEREAL_NVP(outputHeight));
  ar(CEREAL_NVP(poolingLayer));
}

} // namespace ann
} // namespace mlpack

#endif
