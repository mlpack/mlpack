/**
 * @file methods/ann/layer/recurrent_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Recurrent class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP

// In case it hasn't yet been included.
#include "recurrent.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
Recurrent<InputType, OutputType>::Recurrent() :
    rho(0),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
template<
    typename StartModuleType,
    typename InputModuleType,
    typename FeedbackModuleType,
    typename TransferModuleType
>
Recurrent<InputType, OutputType>::Recurrent(
    const StartModuleType& start,
    const InputModuleType& input,
    const FeedbackModuleType& feedback,
    const TransferModuleType& transfer,
    const size_t rho) :
    startModule(new StartModuleType(start)),
    inputModule(new InputModuleType(input)),
    feedbackModule(new FeedbackModuleType(feedback)),
    transferModule(new TransferModuleType(transfer)),
    rho(rho),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0)
{
  initialModule = new SequentialType<InputType, OutputType>();
  mergeModule = new AddMerge<InputType, OutputType>(false, false, false);
  recurrentModule = new SequentialType<InputType, OutputType>(false, false);

  initialModule->Add(inputModule);
  initialModule->Add(startModule);
  initialModule->Add(transferModule);

  mergeModule->Add(inputModule);
  mergeModule->Add(feedbackModule);

  recurrentModule->Add(mergeModule);
  recurrentModule->Add(transferModule);

  network.push_back(initialModule);
  network.push_back(mergeModule);
  network.push_back(feedbackModule);
  network.push_back(recurrentModule);
}

template<typename InputType, typename OutputType>
Recurrent<InputType, OutputType>::Recurrent(
    const Recurrent& network) :
    rho(network.rho),
    forwardStep(network.forwardStep),
    backwardStep(network.backwardStep),
    gradientStep(network.gradientStep)
{
  startModule = network.startModule->Clone();
  inputModule = network.inputModule->Clone();
  feedbackModule = network.feedbackModule->Clone();
  transferModule = network.transferModule->Clone();

  initialModule = new SequentialType<InputType, OutputType>();
  mergeModule = new AddMerge<InputType, OutputType>(false, false, false);
  recurrentModule = new SequentialType<InputType, OutputType>(false, false);

  initialModule->Add(inputModule);
  initialModule->Add(startModule);
  initialModule->Add(transferModule);

  mergeModule->Add(inputModule);
  mergeModule->Add(feedbackModule);

  recurrentModule->Add(mergeModule);
  recurrentModule->Add(transferModule);

  this->network.push_back(initialModule);
  this->network.push_back(mergeModule);
  this->network.push_back(feedbackModule);
  this->network.push_back(recurrentModule);
}

template<typename InputType, typename OutputType>
void Recurrent<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  InitializeForwardPassMemory();

  // Convenience names.
  OutputType& inputOutput = layerOutputs[0];
  OutputType& mergeOutput = layerOutputs[1];
  OutputType& feedbackOutput = layerOutputs[2];
  OutputType& recurrentOutput = layerOutputs[3];

  if (forwardStep == 0)
  {
    initialModule->Forward(input, output);
  }
  else
  {
    inputModule->Forward(input, inputOutput);
    // TODO: how to get transferModule output?
    feedbackModule->Forward(transferModule->OutputParameter(),
        feedbackOutput);
    recurrentModule->Forward(input, output);
  }

  // TODO: how to get transferModule output?
  output = transferModule->OutputParameter();

  // Save the feedback output parameter when training the module.
  if (this->training)
  {
    feedbackOutputParameter.push_back(output);
  }

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    backwardStep = 0;

    if (!recurrentError.is_empty())
    {
      recurrentError.zeros();
    }
  }
}

template<typename InputType, typename OutputType>
void Recurrent<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  InitializeBackwardPassMemory();

  // Convenience names.
  OutputType& inputOutput = layerOutputs[0];
  OutputType& mergeOutput = layerOutputs[1];
  OutputType& feedbackOutput = layerOutputs[2];
  OutputType& recurrentOutput = layerOutputs[3];
  OutputType& inputDelta = layerDeltas[0];
  OutputType& mergeDelta = layerDeltas[1];
  OutputType& feedbackDelta = layerDeltas[2];
  OutputType& recurrentDelta = layerDeltas[3];

  if (!recurrentError.is_empty())
  {
    recurrentError += gy;
  }
  else
  {
    recurrentError = gy;
  }

  if (backwardStep < (rho - 1))
  {
    recurrentModule->Backward(recurrentOutput, recurrentError, recurrentDelta);
    inputModule->Backward(inputOutput, recurrentDelta, g);
    feedbackModule->Backward(feedbackOutput, recurrentDelta, feedbackDelta);
  }
  else
  {
    // TODO: how to get these parameters?
    initialModule->Backward(initialModule->OutputParameter(), recurrentError,
        g);
  }

  recurrentError = feedbackDelta;
  backwardStep++;
}

template<typename InputType, typename OutputType>
void Recurrent<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */)
{
  // Convenience names.
  OutputType& inputOutput = layerOutputs[0];
  OutputType& mergeOutput = layerOutputs[1];
  OutputType& feedbackOutput = layerOutputs[2];
  OutputType& recurrentOutput = layerOutputs[3];
  OutputType& inputDelta = layerDeltas[0];
  OutputType& mergeDelta = layerDeltas[1];
  OutputType& feedbackDelta = layerDeltas[2];
  OutputType& recurrentDelta = layerDeltas[3];
  OutputType& inputGradient = layerGradients[0];
  OutputType& mergeGradient = layerGradients[1];
  OutputType& feedbackGradient = layerGradients[2];
  OutputType& recurrentGradient = layerGradients[3];

  if (gradientStep < (rho - 1))
  {
    recurrentModule->Gradient(input, error, recurrentGradient);
    inputModule->Gradient(input, mergeDelta, inputGradient);
    feedbackModule->Gradient(
        feedbackOutputParameter[feedbackOutputParameter.size() - 2 -
        gradientStep], mergeDelta, feedbackGradient);
  }
  else
  {
    recurrentGradient.zeros();
    inputGradient.zeros();
    feedbackGradient.zeros();

    // TODO: how to do this?
    initialModule->Gradient(input, startModule->Delta(),
        initialModule->Gradient());
  }

  gradientStep++;
  if (gradientStep == rho)
  {
    gradientStep = 0;
    feedbackOutputParameter.clear();
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void Recurrent<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<InputType, OutputType>>(this));

  // TODO: overhaul this

  ar(CEREAL_POINTER(startModule));
  ar(CEREAL_POINTER(inputModule));
  ar(CEREAL_POINTER(feedbackModule));
  ar(CEREAL_POINTER(transferModule));
  ar(CEREAL_NVP(rho));

  // Set up the network.
  if (cereal::is_loading<Archive>())
  {
    initialModule = new SequentialType<InputType, OutputType>();
    mergeModule = new AddMerge<InputType, OutputType>(false, false, false);
    recurrentModule = new SequentialType<InputType, OutputType>(false, false);

    initialModule->Add(inputModule);
    initialModule->Add(startModule);
    initialModule->Add(transferModule);

    mergeModule->Add(inputModule);
    mergeModule->Add(feedbackModule);

    recurrentModule->Add(mergeModule);
    recurrentModule->Add(transferModule);

    network.push_back(initialModule);
    network.push_back(mergeModule);
    network.push_back(feedbackModule);
    network.push_back(recurrentModule);
  }
}

} // namespace mlpack

#endif
