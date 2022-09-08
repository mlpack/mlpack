/**
 * @file methods/ann/layer/recurrent_layer_impl.hpp
 * @author Ryan Curtin
 *
 * Base layer for recurrent neural network layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with the mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_IMPL_HPP

// In case it hasn't been included yet.
#include "recurrent_layer.hpp"

namespace mlpack {

template<typename MatType>
RecurrentLayer<MatType>::RecurrentLayer() :
    Layer<MatType>(),
    currentStep(0),
    previousStep(0)
{ /* Nothing to do. */ }

template<typename MatType>
RecurrentLayer<MatType>::RecurrentLayer(const RecurrentLayer& other) :
    Layer<MatType>(other),
    currentStep(other.currentStep),
    previousStep(other.previousStep)
{ /* Nothing else to do. */ }

template<typename MatType>
RecurrentLayer<MatType>::RecurrentLayer(RecurrentLayer&& other) :
    Layer<MatType>(std::move(other)),
    currentStep(std::move(other.currentStep)),
    previousStep(std::move(other.previousStep))
{ /* Nothing else to do. */ }

template<typename MatType>
RecurrentLayer<MatType>&
RecurrentLayer<MatType>::operator=(const RecurrentLayer& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    currentStep = other.currentStep;
    previousStep = other.previousStep;
  }

  return *this;
}

template<typename MatType>
RecurrentLayer<MatType>&
RecurrentLayer<MatType>::operator=(RecurrentLayer&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    currentStep = std::move(other.currentStep);
    previousStep = std::move(other.previousStep);
  }

  return *this;
}

template<typename MatType>
template<typename Archive>
void RecurrentLayer<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(currentStep));
  ar(CEREAL_NVP(previousStep));
}

} // namespace mlpack

#endif
