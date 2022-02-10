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
    const bool run) :
    run(run), ownsLayers(true)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
AddMerge<InputType, OutputType>::AddMerge(
    const bool run, const bool ownsLayers) :
    run(run), ownsLayers(ownsLayers)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
AddMerge<InputType, OutputType>::~AddMerge()
{

}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  this->InitializeForwardPassMemory();

  if (run)
  {
    for (size_t i = 0; i < this->network.size(); ++i)
    {
      this->network[i]->Forward(input, this->layerOutputs[i]);
    }
  }

  output = this->layerOutputs.front();
  for (size_t i = 1; i < this->network.size(); ++i)
  {
    output += this->layerOutputs[i];
  }
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  this->InitializeBackwardPassMemory();

  if (run)
  {
    for (size_t i = 0; i < this->network.size(); ++i)
    {
      this->network[i]->Backward(this->layerOutputs[i], gy,
          this->layerDeltas[i]);
    }

    g = this->layerDeltas[0];
    for (size_t i = 1; i < this->network.size(); ++i)
    {
      g += this->layerDeltas[i];
    }
  }
  else
  {
    g = gy;
  }
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g,
    const size_t index)
{
  this->network[index]->Backward(this->layerOutputs[index], gy, g);
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  if (run)
  {
    size_t start = 0;
    for (size_t i = 0; i < this->network.size(); ++i)
    {
      this->network[i]->Gradient(input, error, OutputType(gradient.colptr(start),
          1, this->network[i]->WeightSize(), false, true));
      start += this->network[i]->WeightSize();
    }
  }
}

template<typename InputType, typename OutputType>
void AddMerge<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient,
    const size_t index)
{
  this->network[index]->Gradient(input, error, gradient);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void AddMerge<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(run));
  ar(CEREAL_NVP(ownsLayers));
}

} // namespace ann
} // namespace mlpack

#endif
