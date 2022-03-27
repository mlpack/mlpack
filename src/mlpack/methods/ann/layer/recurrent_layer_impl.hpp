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
namespace ann {

template<typename InputType, typename OutputType>
RecurrentLayer<InputType, OutputType>::RecurrentLayer() :
    Layer<InputType, OutputType>(),
    currentStep(0),
    previousStep(0)
{ /* Nothing to do. */ }

template<typename InputType, typename OutputType>
RecurrentLayer<InputType, OutputType>::RecurrentLayer(
    const RecurrentLayer& other) :
    Layer<InputType, OutputType>(other),
    currentStep(other.currentStep),
    previousStep(other.previousStep)
{ /* Nothing else to do. */ }

template<typename InputType, typename OutputType>
RecurrentLayer<InputType, OutputType>::RecurrentLayer(RecurrentLayer&& other) :
    Layer<InputType, OutputType>(std::move(other)),
    currentStep(std::move(other.currentStep)),
    previousStep(std::move(other.previousStep))
{ /* Nothing else to do. */ }

template<typename InputType, typename OutputType>
RecurrentLayer<InputType, OutputType>&
RecurrentLayer<InputType, OutputType>::operator=(const RecurrentLayer& other)
{
  if (this != &other)
  {
    Layer<InputType, OutputType>::operator=(other);
    currentStep = other.currentStep;
    previousStep = other.previousStep;
  }

  return *this;
}

template<typename InputType, typename OutputType>
RecurrentLayer<InputType, OutputType>&
RecurrentLayer<InputType, OutputType>::operator=(RecurrentLayer&& other)
{
  if (this != &other)
  {
    Layer<InputType, OutputType>::operator=(std::move(other));
    currentStep = std::move(other.currentStep);
    previousStep = std::move(other.previousStep);
  }

  return *this;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void RecurrentLayer<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(currentStep));
  ar(CEREAL_NVP(previousStep));
}

} // namespace ann
} // namespace mlpack

#endif
