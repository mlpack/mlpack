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

template<typename InputDataType, typename OutputDataType>
AdaptiveMaxPooling<InputDataType, OutputDataType>::AdaptiveMaxPooling()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
AdaptiveMaxPooling<InputDataType, OutputDataType>::AdaptiveMaxPooling(
    const size_t outputWidth,
    const size_t outputHeight) :
    AdaptiveMaxPooling(std::tuple<size_t, size_t>(outputWidth, outputHeight))
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
AdaptiveMaxPooling<InputDataType, OutputDataType>::AdaptiveMaxPooling(
    const std::tuple<size_t, size_t>& outputShape):
    outputWidth(std::get<0>(outputShape)),
    outputHeight(std::get<1>(outputShape)),
    reset(false)
{
  poolingLayer = ann::MaxPooling<>(0, 0);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void AdaptiveMaxPooling<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  if (!reset)
  {
    IntializeAdaptivePadding();
    reset = true;
  }

  poolingLayer.Forward(input, output);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void AdaptiveMaxPooling<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>& input,
  const arma::Mat<eT>& gy,
  arma::Mat<eT>& g)
{
  poolingLayer.Backward(input, gy, g);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void AdaptiveMaxPooling<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);
  ar & BOOST_SERIALIZATION_NVP(reset);

  if (version > 0)
    ar & BOOST_SERIALIZATION_NVP(poolingLayer);
}

} // namespace ann
} // namespace mlpack

#endif
