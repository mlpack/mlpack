/**
 * @file methods/ann/layer/recurrent_attention_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the RecurrentAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_IMPL_HPP

// In case it hasn't yet been included.
#include "recurrent_attention.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
RecurrentAttention<InputType, OutputType>::RecurrentAttention() :
    outSize(0),
    rho(0),
    forwardStep(0),
    backwardStep(0),
    deterministic(false)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
template<typename RNNModuleType, typename ActionModuleType>
RecurrentAttention<InputType, OutputType>::RecurrentAttention(
    const size_t outSize,
    const RNNModuleType& rnn,
    const ActionModuleType& action,
    const size_t rho) :
    outSize(outSize),
    rnnModule(new RNNModuleType(rnn)),
    actionModule(new ActionModuleType(action)),
    rho(rho),
    forwardStep(0),
    backwardStep(0),
    deterministic(false)
{
  network.push_back(rnnModule);
  network.push_back(actionModule);
}

template<typename InputType, typename OutputType>
void RecurrentAttention<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  InitializeForwardPassMemory();

  // Convenience naming.
  OutputType& rnnOutput = layerOutputs.front();
  OutputType& actionOutput = layerOutputs.back();

  // Initialize the action input.
  if (initialInput.is_empty())
  {
    initialInput = zeros(outSize, input.n_cols);
  }

  // Propagate through the action and recurrent module.
  for (forwardStep = 0; forwardStep < rho; ++forwardStep)
  {
    if (forwardStep == 0)
    {
      actionModule->Forward(initialInput, actionOutput);
    }
    else
    {
      actionModule->Forward(rnnOutput, actionOutput);
    }

    // Initialize the glimpse input.
    InputType glimpseInput = zeros(input.n_elem, 2);
    glimpseInput.col(0) = input;
    glimpseInput.submat(0, 1, actionOutput.n_elem - 1, 1) =
        actionOutput;

    rnnModule->Forward(glimpseInput, rnnOutput);

    // Save the output parameter when training the module.
    if (!deterministic)
    {
      for (size_t l = 0; l < network.size(); ++l)
      {
        // TODO: what if network[i] has a Model()?
        // TODO: what does this actually do?  do we need it?
        moduleOutputParameter.push_back(network[l]->OutputParameter());
      }
    }
  }

  output = rnnOutput;

  forwardStep = 0;
  backwardStep = 0;
}

template<typename InputType, typename OutputType>
void RecurrentAttention<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  InitializeBackwardPassMemory();

  // Convenience names.
  OutputType& rnnOutput = layerOutputs.front();
  OutputType& actionOutput = layerOutputs.back();
  OutputType& rnnGradient = layerGradients.front();
  OutputType& actionGradient = layerGradients.back();

  if (intermediateGradient.is_empty() && backwardStep == 0)
  {
    // Initialize the attention gradients.
    // TODO: do rnnModule or actionModule have a Model()?  We may need to
    // account for those weights too.
    size_t weights = rnnModule->Parameters().n_elem +
        actionModule->Parameters().n_elem;

    intermediateGradient = zeros(weights, 1);
    attentionGradient = zeros(weights, 1);

    // Initialize the action error.
    actionError = zeros(actionOutput.n_rows, actionOutput.n_cols);
  }

  // Propagate the attention gradients.
  if (backwardStep == 0)
  {
    size_t offset = 0;
    // TODO: what if rnnModule has a Model()?
    rnnGradient = OutputType(intermediateGradient.memptr() + offset,
        rnnModule->Parameters().n_rows, rnnModule->Parameters().n_cols, false,
        false);
    offset += rnnModule->Parameters().n_elem;
    actionGradient = OutputType(intermediateGradient.memptr() + offset,
        actionModule->Parameters().n_rows, actionModule->Parameters().n_cols,
        false, false);

    attentionGradient.zeros();
  }

  // Back-propagate through time.
  for (; backwardStep < rho; backwardStep++)
  {
    if (backwardStep == 0)
    {
      recurrentError = gy;
    }
    else
    {
      recurrentError = actionDelta;
    }

    for (size_t l = 0; l < network.size(); ++l)
    {
      // TODO: handle case where HasModelCheck is true
      network[network.size() - 1 - l] = moduleOutputParameter.back();
      moduleOutputParameter.pop_back();
    }

    if (backwardStep == (rho - 1))
    {
      actionModule->Backward(actionOutput, actionError, actionDelta);
    }
    else
    {
      actionModule->Backward(initialInput, actionError, actionDelta);
    }

    rnnModule->Backward(rnnOutput, recurrentError, rnnDelta);

    if (backwardStep == 0)
    {
      g = rnnDelta.col(1);
    }
    else
    {
      g += rnnDelta.col(1);
    }

    IntermediateGradient();
  }
}

template<typename InputType, typename OutputType>
void RecurrentAttention<InputType, OutputType>::Gradient(
    const InputType& /* input */,
    const OutputType& /* error */,
    OutputType& /* gradient */)
{
  // Convenience naming.
  OutputType& rnnGradient = layerGradients.front();
  OutputType& actionGradient = layerGradients.back();

  size_t offset = 0;
  // TODO: handle case where rnnModule or actionModule have a model
  if (rnnModule->Parameters().n_elem != 0)
  {
    rnnGradient = attentionGradient.submat(offset, 0, offset +
        rnnModule->Parameters().n_elem - 1, 0);
    offset += rnnModule->Parameters().n_elem;
  }

  if (actionModule->Parameters().n_elem != 0)
  {
    actionGradient = attentionGradient.submat(offset, 0, offset +
        actionModule->Parameters().n_elem - 1, 0);
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void RecurrentAttention<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(rho));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(forwardStep));
  ar(CEREAL_NVP(backwardStep));

  // TODO: lots of clearing?
}

} // namespace mlpack

#endif
