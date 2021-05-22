/**
 * @file methods/ann/layer/add_merge_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the AddMerge module which accumulates the output of the given
 * modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP

// In case it hasn't yet been included.
#include "add_merge.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
AddMerge<InputType, OutputType>::AddMerge(
    const bool model, const bool run) :
    model(model), run(run), ownsLayers(!model)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
AddMerge<InputType, OutputType>::AddMerge(
    const bool model, const bool run, const bool ownsLayers) :
    model(model), run(run), ownsLayers(ownsLayers)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
AddMerge<InputType, OutputType>::~AddMerge()
{
  if (!model && ownsLayers)
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Forward(input, network->OutputParameter());
    }
  }

  output = network.front()->OutputParameter();
  for (size_t i = 1; i < network.size(); ++i)
  {
    output += network[i]->OutputParameter();
  }
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Backward(network[i]->OutputParameter(), gy,
          network[i]->Delta());
    }

    g = network[0]->Delta();
    for (size_t i = 1; i < network.size(); ++i)
    {
      g += network[i]->Delta();
    }
  }
  else
    g = gy;
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g,
    const size_t index)
{
  network[index]->Backward(network[index]->OutputParameter(), gy,
      network[index]->Delta());
  g = network[index]->Delta();
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */ )
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Gradient(input, error, network[i]->Gradient());
    }
  }
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */,
    const size_t index)
{
  network[index]->Gradient(input, error, network[index]->Gradient());
}

template<typename InputType, typename OutputType>
template<typename Archive>
void AddMerge<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_VECTOR_POINTER(network));
  ar(CEREAL_NVP(model));
  ar(CEREAL_NVP(run));
  ar(CEREAL_NVP(ownsLayers));
  ar(CEREAL_NVP(weights));
}

} // namespace ann
} // namespace mlpack

#endif
