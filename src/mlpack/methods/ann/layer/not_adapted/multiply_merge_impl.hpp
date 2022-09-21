/**
 * @file methods/ann/layer/multiply_merge_impl.hpp
 * @author Haritha Nair
 *
 * Definition of the MultiplyMerge module which multiplies the output of the
 * given modules element-wise.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_IMPL_HPP

// In case it hasn't yet been included.
#include "multiply_merge.hpp"

// #include "../visitor/forward_visitor.hpp"
// #include "../visitor/backward_visitor.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
MultiplyMergeType<InputType, OutputType>::MultiplyMergeType(
    const bool run) :
    run(run)
{
  // Nothing to do here.
}

// TODO: is this destructor needed?
template<typename InputType, typename OutputType>
MultiplyMergeType<InputType, OutputType>::~MultiplyMergeType()
{
  for (size_t i = 0; i < network.size(); ++i)
    delete network[i];
}

template<typename InputType, typename OutputType>
void MultiplyMergeType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  InitializeForwardPassMemory();

  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Forward(input, layerOutputs[i]);
    }
  }

  output = layerOutputs.front();
  for (size_t i = 1; i < network.size(); ++i)
  {
    output %= layerOutputs[i];
  }
}

template<typename InputType, typename OutputType>
void MultiplyMergeType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  InitializeBackwardPassMemory();

  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Backward(layerOutputs[i], gy, layerDeltas[i]);
    }

    g = layerDeltas.front();
    for (size_t i = 1; i < network.size(); ++i)
    {
      g += layerDeltas[i];
    }
  }
  else
  {
    g = gy;
  }
}

template<typename InputType, typename OutputType>
void MultiplyMergeType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */ )
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Gradient(input, error, layerGradients[i]);
    }
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MultiplyMergeType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(run));
}

} // namespace mlpack

#endif
