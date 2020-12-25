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
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
AdaptiveMaxPoolingType<InputType, OutputType>::AdaptiveMaxPoolingType()
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
AdaptiveMaxPoolingTypeType<InputType, OutputType>::AdaptiveMaxPoolingType(
    const size_t outputWidth,
    const size_t outputHeight) :
    AdaptiveMaxPoolingType(std::tuple<size_t, size_t>(outputWidth, outputHeight))
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
AdaptiveMaxPoolingType<InputType, OutputType>::AdaptiveMaxPoolingType(
    const std::tuple<size_t, size_t>& outputShape):
    outputWidth(std::get<0>(outputShape)),
    outputHeight(std::get<1>(outputShape)),
    reset(false)
{
  poolingLayer = ann::MaxPooling<>(0, 0);
}

template<typename InputType, typename OutputType>
void AdaptiveMaxPoolingType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (!reset)
  {
    IntializeAdaptivePadding();
    reset = true;
  }

  poolingLayer.Forward(input, output);
}

template<typename InputType, typename OutputType>
void AdaptiveMaxPoolingType<InputType, OutputType>::Backward(
  const InputType& input,
  const OutputType& gy,
  OutputType& g)
{
  poolingLayer.Backward(input, gy, g);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void AdaptiveMaxPoolingType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(outputWidth));
  ar(CEREAL_NVP(outputHeight));
  ar(CEREAL_NVP(reset));
  ar(CEREAL_NVP(poolingLayer));
}

} // namespace ann
} // namespace mlpack

#endif
