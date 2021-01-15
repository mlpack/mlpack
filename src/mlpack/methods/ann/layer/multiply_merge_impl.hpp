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
// #include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
MultiplyMergeType<InputType, OutputType>::MultiplyMergeType(
    const bool model, const bool run) :
    model(model), run(run), ownsLayer(!model)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
MultiplyMergeType<InputType, OutputType>::~MultiplyMergeType()
{
  if (ownsLayer)
  {
    // std::for_each(network.begin(), network.end(),
    //     boost::apply_visitor(deleteVisitor));
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }
}

template<typename InputType, typename OutputType>
void MultiplyMergeType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      // boost::apply_visitor(ForwardVisitor(input,
      //     boost::apply_visitor(outputParameterVisitor, network[i])),
      //     network[i]);
      network[i]->Forward(input, network[i]->OutputParameter());
    }
  }

  // output = boost::apply_visitor(outputParameterVisitor, network.front());
  output = network.front()->OutputParameter();
  for (size_t i = 1; i < network.size(); ++i)
  {
    // output %= boost::apply_visitor(outputParameterVisitor, network[i]);
    output %= network[i]->OutputParameter();
  }
}

template<typename InputType, typename OutputType>
void MultiplyMergeType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      // boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      //     outputParameterVisitor, network[i]), gy,
      //     boost::apply_visitor(deltaVisitor, network[i])), network[i]);
      network[i]->Backward(network[i]->OutputParameter(),
                           gy,
                           network[i]->Delta());
    }

    // g = boost::apply_visitor(deltaVisitor, network[0]);
    g = network[0]->Delta();
    for (size_t i = 1; i < network.size(); ++i)
    {
      // g += boost::apply_visitor(deltaVisitor, network[i]);
      g += network[i]->Delta();
    }
  }
  else
    g = gy;
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
      // boost::apply_visitor(GradientVisitor(input, error), network[i]);
      network[i]->Gradient(input, error, network[i]->Gradient());
    }
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MultiplyMergeType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // Be sure to clear other layers before loading.
  if (cereal::is_loading<Archive>())
    network.clear();

  // ar(CEREAL_VECTOR_VARIANT_POINTER(network));
  ar(CEREAL_NVP(model));
  ar(CEREAL_NVP(run));
  ar(CEREAL_NVP(ownsLayer));
}

} // namespace ann
} // namespace mlpack

#endif
