/**
 * @file methods/ann/layer/highway_impl.hpp
 * @author Konstantin Sidorov
 * @author Saksham Bansal
 *
 * Implementation of Highway layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HIGHWAY_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_HIGHWAY_IMPL_HPP

// In case it hasn't yet been included.
#include "highway.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::HighwayType()
{
  // Nothing to do here.
  // TODO: how do we add the child layers ?? (read paper ...)
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::~HighwayType()
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::HighwayType(const HighwayType& other) :
    MultiLayer<InputType, OutputType>(other)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::HighwayType(HighwayType&& other) :
    MultiLayer<InputType, OutputType>(std::move(other))
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>&
HighwayType<InputType, OutputType>::operator=(const HighwayType& other)
{
  if (&other == this)
  {
    MultiLayer<InputType, OutputType>::operator=(other);
  }

  return *this;
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>&
HighwayType<InputType, OutputType>::operator=(HighwayType&& other)
{
  if (&other == this)
  {
    MultiLayer<InputType, OutputType>::operator=(std::move(other));
  }

  return *this;
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  transformWeight = OutputType(weightsPtr, this->inSize,
      this->inSize, false, false);
  transformBias = OutputType(weightsPtr + transformWeight.n_elem,
      this->inSize, 1, false, false);

  size_t start = transformWeight.n_elem + transformBias.n_elem;
  for (size_t i = 0; i < this->network.size(); ++i)
  {
    this->network[i]->SetWeights(weightsPtr + start);
    start += this->network[i]->WeightSize();
  }
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  this->InitializeForwardPassMemory(input.n_cols);

  this->network.front()->Forward(input, this->layerOutputs.front());

  for (size_t i = 1; i < this->network.size(); ++i)
  {
    this->network[i]->Forward(this->layerOutputs[i - 1], this->layerOutputs[i]);
  }

  output = this->layerOutputs.back(); // TODO: can this be cleaned up?

  // TODO: move to ComputeOutputDimensions()
  if (arma::size(output) != arma::size(input))
  {
    Log::Fatal << "The sizes of the output and input matrices of the Highway"
        << " network should be equal. Please examine the network layers.";
  }

  transformGate = transformWeight * input;
  transformGate.each_col() += transformBias;
  transformGateActivation = 1.0 /(1 + exp(-transformGate));
  output = (this->layerOutputs.back() % transformGateActivation) +
      (input % (1 - transformGateActivation));
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Backward(
    const InputType& input,
    const OutputType& gy,
    OutputType& g)
{
  this->InitializeBackwardPassMemory(input.n_cols);

  OutputType gyTransform = gy % transformGateActivation;
  this->network.back()->Backward(this->layerOutputs.back(), gyTransform,
      this->layerDeltas.back());

  for (size_t i = 2; i < this->network.size() + 1; ++i)
  {
    this->network[this->network.size() - i]->Backward(
        this->layerOutputs[this->network.size() - i],
        this->layerDeltas[this->network.size() - i + 1],
        this->layerDeltas[this->network.size() - i]);
  }

  transformGateError = gy % (gy - input) %
      transformGateActivation % (1.0 - transformGateActivation);
  g = this->layerDeltas.front() + (transformWeight.t() * transformGateError) +
      (gy % (1 - transformGateActivation));
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  // Create an alias for the gradient that only refers to the elements in the
  // network itself.
  OutputType layerGradient(gradient.memptr() + (this->inSize *
      (this->inSize + 1)), 1, gradient.n_elem - (this->inSize *
      (this->inSize + 1)), false, true);
  this->InitializeGradientPassMemory(layerGradient);

  OutputType errorTransform = error % transformGateActivation;
  this->network.back()->Gradient(
      this->layerOutputs[this->network.size() - 2],
      errorTransform,
      this->layerGradients[this->network.size() - 1]);

  for (size_t i = 2; i < this->network.size(); ++i)
  {
    this->network[this->network.size() - i]->Gradient(
        this->layerOutputs[this->network.size() - i - 1],
        this->layerDeltas[this->network.size() - i],
        this->layerGradients[this->network.size() - i]);
  }

  this->network.front()->Gradient(
      input,
      this->layerDeltas[1],
      this->layerGradients.front());

  gradient.submat(0, 0, transformWeight.n_elem - 1, 0) = vectorise(
      transformGateError * input.t());
  gradient.submat(transformWeight.n_elem, 0, transformWeight.n_elem +
      transformBias.n_elem - 1, 0) = sum(transformGateError, 1);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void HighwayType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));
}

} // namespace mlpack

#endif
