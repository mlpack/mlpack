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

#include "../visitor/add_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"
#include "../visitor/gradient_zero_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
Recurrent<InputType, OutputType>::Recurrent() :
    rho(0),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false),
    // TODO: does ownsLayer do anything?
    ownsLayer(false)
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
    gradientStep(0),
    deterministic(false),
    ownsLayer(true)
{
  initialModule = new Sequential<>();
  mergeModule = new AddMerge<>(false, false, false);
  recurrentModule = new Sequential<>(false, false);

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
    gradientStep(network.gradientStep),
    deterministic(network.deterministic),
    ownsLayer(network.ownsLayer)
{
  startModule = network.startModule->Clone();
  inputModule = network.inputModule->Clone();
  feedbackModule = network.feedbackModule->Clone();
  transferModule = network.transferModule->Clone();

  initialModule = new Sequential<>();
  mergeModule = new AddMerge<>(false, false, false);
  recurrentModule = new Sequential<>(false, false);

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
void Recurrent<InputType, OutputType, CustomLayers...>::Forward(
    const InputType& input, OutputType& output)
{
  if (forwardStep == 0)
  {
    initialModule->Forward(input, output);
  }
  else
  {
    inputModule->Forward(input, inputModule->OutputParameter());
    feedbackModule->Forward(transferModule->OutputParameter(),
        feedbackModule->OutputParameter());
    recurrentModule->Forward(input, output);
  }

  output = transferModule->OutputParameter();

  // Save the feedback output parameter when training the module.
  if (!deterministic)
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

template<typename InputType, typename OutputType,
         typename... CustomLayers>
void Recurrent<InputType, OutputType, CustomLayers...>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
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
    recurrentModule->Backward(recurrentModule->OutputParameter(),
        recurrentError, recurrentModule->Delta());
    inputModule->Backward(inputModule->OutputParameter(),
        recurrentModule->Delta(), g);
    feedbackModule->Backward(feedbackModule->OutputParameter(),
        recurrentModule->Delta(), feedbackModule->Delta());
  }
  else
  {
    initialModule->Backward(initialModule->OutputParameter(), recurrentError,
        g);
  }

  recurrentError = feedbackModule->Delta();
  backwardStep++;
}

template<typename InputType, typename OutputType,
         typename... CustomLayers>
void Recurrent<InputType, OutputType, CustomLayers...>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */)
{
  if (gradientStep < (rho - 1))
  {
    recurrentModule->Gradient(input, error, recurrentModule->Gradient());
    inputModule->Gradient(input, mergeModule->Delta(), inputModule->Gradient());
    feedbackModule->Gradient(
        feedbackOutputParameter[feedbackOutputParameter.size() - 2 -
        gradientStep], mergeModule->Delta(), feedbackModule->Gradient());
  }
  else
  {
    // TODO: what about if there is no Gradient()?
    recurrentModule->Gradient().zeros();
    inputModule->Gradient().zeros();
    feedbackModule->Gradient().zeros();

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

template<typename InputType, typename OutputType,
         typename... CustomLayers>
template<typename Archive>
void Recurrent<InputType, OutputType, CustomLayers...>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  // Clean up memory, if we are loading.
  if (cereal::is_loading<Archive>())
  {
    // TODO: should we consider ownsLayer here?

    // Clear old things, if needed.
    network.clear();
  }

  ar(CEREAL_POINTER(startModule));
  ar(CEREAL_POINTER(inputModule));
  ar(CEREAL_POINTER(feedbackModule));
  ar(CEREAL_POINTER(transferModule));
  ar(CEREAL_NVP(rho));
  ar(CEREAL_NVP(ownsLayer));

  // Set up the network.
  if (cereal::is_loading<Archive>())
  {
    initialModule = new Sequential<>();
    mergeModule = new AddMerge<>(false, false, false);
    recurrentModule = new Sequential<>(false, false);

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

} // namespace ann
} // namespace mlpack

#endif
