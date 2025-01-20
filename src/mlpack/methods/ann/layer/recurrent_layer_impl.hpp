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
    Layer<MatType>()
{ /* Nothing to do. */ }

template<typename MatType>
RecurrentLayer<MatType>::RecurrentLayer(const RecurrentLayer& other) :
    Layer<MatType>(other)
{ /* Nothing else to do. */ }

template<typename MatType>
RecurrentLayer<MatType>::RecurrentLayer(RecurrentLayer&& other) :
    Layer<MatType>(std::move(other))
{ /* Nothing else to do. */ }

template<typename MatType>
RecurrentLayer<MatType>&
RecurrentLayer<MatType>::operator=(const RecurrentLayer& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
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
  }

  return *this;
}

template<typename MatType>
void RecurrentLayer<MatType>::ClearRecurrentState(
    const size_t bpttSteps,
    const size_t batchSize)
{
  const size_t numRecurrentElements = this->RecurrentSize();
  if (bpttSteps == 0)
  {
    // In this case we are only doing forward mode, so only allocate space for
    // storage of one time step.
    recurrentState.set_size(numRecurrentElements, batchSize, 1);
    recurrentGradient.reset();
  }
  else
  {
    recurrentState.set_size(numRecurrentElements, batchSize, bpttSteps);
    // Backward() and Gradient() only ever need the current and the previous
    // time steps; so, we can save some space by using only two.
    recurrentGradient.set_size(numRecurrentElements, batchSize,
        std::min((size_t) 2, bpttSteps));
  }
}

template<typename MatType>
void RecurrentLayer<MatType>::CurrentStep(const size_t& step, const bool end)
{
  currentStep = step;
  atFinalStep = end;
}

template<typename MatType>
const MatType& RecurrentLayer<MatType>::RecurrentState(const size_t t) const
{
  // Recurrent state is stored in a circular buffer.
  return recurrentState.slice(t % recurrentState.n_slices);
}

template<typename MatType>
MatType& RecurrentLayer<MatType>::RecurrentState(const size_t t)
{
  // Recurrent state is stored in a circular buffer.
  return recurrentState.slice(t % recurrentState.n_slices);
}

template<typename MatType>
const MatType& RecurrentLayer<MatType>::RecurrentGradient(const size_t t) const
{
  // Recurrent gradients are stored in a circular buffer.
  return recurrentGradient.slice(t % recurrentGradient.n_slices);
}

template<typename MatType>
MatType& RecurrentLayer<MatType>::RecurrentGradient(const size_t t)
{
  // Recurrent gradients are stored in a circular buffer.
  return recurrentGradient.slice(t % recurrentGradient.n_slices);
}

template<typename MatType>
template<typename Archive>
void RecurrentLayer<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  // No state needs to be serialized, because it will all be reset on the next
  // forward pass.
}

} // namespace mlpack

#endif
