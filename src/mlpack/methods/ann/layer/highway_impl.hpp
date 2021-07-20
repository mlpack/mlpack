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
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::HighwayType() :
    inSize(0),
    reset(false),
    width(0),
    height(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::HighwayType(
    const size_t inSize) :
    inSize(inSize),
    reset(false),
    width(0),
    height(0)
{
  weights.set_size(inSize * inSize + inSize, 1);
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::~HighwayType()
{
  for (size_t i = 0; i < network.size(); ++i)
  {
    if (networkOwnerships[i])
      delete network[i];
  }
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  transformWeight = OutputType(weightsPtr, inSize, inSize, false, false);
  transformBias = OutputType(weightsPtr + transformWeight.n_elem,
      inSize, 1, false, false);

  size_t start = transformWeight.n_elem + transformBias.n_elem;
  for (size_t i = 0; i < network.size(); ++i)
  {
    network[i]->SetWeights(weightsPtr + start);
    start += network[i]->WeightSize();
  }
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  InitializeForwardPassMemory();

  network.front()->Forward(input, layerOutputs.front());

  for (size_t i = 1; i < network.size(); ++i)
  {
    network[i]->Forward(layerOutputs[i - 1], layerOutputs[i]);
  }

  output = network.back()->OutputParameter();

  if (arma::size(output) != arma::size(input))
  {
    Log::Fatal << "The sizes of the output and input matrices of the Highway"
        << " network should be equal. Please examine the network layers.";
  }

  transformGate = transformWeight * input;
  transformGate.each_col() += transformBias;
  transformGateActivation = 1.0 /(1 + arma::exp(-transformGate));
  inputParameter = input;
  networkOutput = output; // TODO: what is done with this?
  output = (layerOutputs.back() % transformGateActivation) +
      (input % (1 - transformGateActivation));
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  InitializeBackwardPassMemory();

  OutputType gyTransform = gy % transformGateActivation;
  network.back()->Backward(layerOutputs.back(), gyTransform,
      layerDeltas.back());

  for (size_t i = 2; i < network.size() + 1; ++i)
  {
    network[network.size() - i]->Backward(layerOutputs[network.size() - i],
        layerDeltas[network.size() - i + 1], layerDeltas[network.size() - i]);
  }

  transformGateError = gy % (networkOutput - inputParameter) %
      transformGateActivation % (1.0 - transformGateActivation);
  g = layerDeltas.front() + (transformWeight.t() * transformGateError) +
      (gy % (1 - transformGateActivation));
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  OutputType errorTransform = error % transformGateActivation;
  size_t gradientStart = gradient.n_elem -
      network[network.size() - 1].WeightSize();
  network.back()->Gradient(
      layerOutputs[network.size() - 2],
      errorTransform,
      OutputType(gradient.colptr(gradientStart), 1,
          network[network.size() - 1].WeightSize(), false, true)
    );

  for (size_t i = 2; i < network.size(); ++i)
  {
    gradientStart -= network[network.size() - i]->WeightSize();
    network[network.size() - i]->Gradient(
        layerOutputs[network.size() - i - 1],
        layerDeltas[network.size() - i],
        OutputType(gradient.colptr(gradientStart), 1,
            network[network.size() - i]->WeightSize(), false, true)
      );
  }

  network.front()->Gradient(
      input,
      layerDeltas[1],
      layerDeltas.front()
    );

  gradient.submat(0, 0, transformWeight.n_elem - 1, 0) = arma::vectorise(
      transformGateError * input.t());
  gradient.submat(transformWeight.n_elem, 0, transformWeight.n_elem +
      transformBias.n_elem - 1, 0) = arma::sum(transformGateError, 1);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void HighwayType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_VECTOR_POINTER(network));

  // Reset the memory.
  if (Archive::is_loading::value)
  {
    networkOwnerships.clear();
    networkOwnerships.resize(network.size(), true);
  }
}

} // namespace ann
} // namespace mlpack

#endif
