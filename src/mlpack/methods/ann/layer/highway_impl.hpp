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
    model(true),
    reset(false),
    width(0),
    height(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::HighwayType(
    const size_t inSize,
    const bool model) :
    inSize(inSize),
    model(model),
    reset(false),
    width(0),
    height(0)
{
  weights.set_size(inSize * inSize + inSize, 1);
}

template<typename InputType, typename OutputType>
HighwayType<InputType, OutputType>::~HighwayType()
{
  if (!model)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      if (networkOwnerships[i])
        delete network[i];
    }
  }
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Reset()
{
  transformWeight = OutputType(weights.memptr(), inSize, inSize, false, false);
  transformBias = OutputType(weights.memptr() + transformWeight.n_elem,
      inSize, 1, false, false);
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  network.front()->Forward(input, network.front()->OutputParameter());

  // if (!reset)
  // {
  //   if (boost::apply_visitor(outputWidthVisitor, network.front()) != 0)
  //   {
  //     width = boost::apply_visitor(outputWidthVisitor, network.front());
  //   }

  //   if (boost::apply_visitor(outputHeightVisitor, network.front()) != 0)
  //   {
  //     height = boost::apply_visitor(outputHeightVisitor, network.front());
  //   }
  // }

  for (size_t i = 1; i < network.size(); ++i)
  {
    // if (!reset)
    // {
    //   // Set the input width.
    //   boost::apply_visitor(SetInputWidthVisitor(width), network[i]);

    //   // Set the input height.
    //   boost::apply_visitor(SetInputHeightVisitor(height), network[i]);
    // }

    network[i]->Forward(
        network[i - 1]->OutputParameter(),
        network[i]->OutputParameter()
      );

    // if (!reset)
    // {
    //   // Get the output width.
    //   if (boost::apply_visitor(outputWidthVisitor, network[i]) != 0)
    //   {
    //     width = boost::apply_visitor(outputWidthVisitor, network[i]);
    //   }

    //   // Get the output height.
    //   if (boost::apply_visitor(outputHeightVisitor, network[i]) != 0)
    //   {
    //     height = boost::apply_visitor(outputHeightVisitor, network[i]);
    //   }
    // }
  }
  // if (!reset)
  // {
  //   reset = true;
  // }

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
  networkOutput = output;
  output = (output % transformGateActivation) +
      (input % (1 - transformGateActivation));
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  OutputType gyTransform = gy % transformGateActivation;
  network.back()->Backward(network.back()->OutputParameter(),
                           gyTransform,
                           network.back()->Delta());

  for (size_t i = 2; i < network.size() + 1; ++i)
  {
    network[network.size() - i]->Backward(
        network[network.size() - i]->OutputParameter(),
        network[network.size() - i + 1]->Delta(),
        network[network.size() - i]->Delta()
      );
  }

  g = network.front()->Delta();

  transformGateError = gy % (networkOutput - inputParameter) %
      transformGateActivation % (1.0 - transformGateActivation);
  g += transformWeight.t() * transformGateError;
  g += gy % (1 - transformGateActivation);
}

template<typename InputType, typename OutputType>
void HighwayType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  OutputType errorTransform = error % transformGateActivation;
  network.back()->Gradient(
      network[network.size() - 2]->OutputParameter(),
      errorTransform,
      network[network.size() - 2]->Gradient()
    );

  for (size_t i = 2; i < network.size(); ++i)
  {
    network[network.size() - i]->Gradient(
        network[network.size() - i - 1]->OutputParameter(),
        network[network.size() - i + 1]->Delta(),
        network[network.size() - i]->Gradient()
      );
  }

  network.front()->Gradient(
      input,
      network[1]->Delta(),
      network.front()->Delta()
    );

  gradient.submat(0, 0, transformWeight.n_elem - 1, 0) = arma::vectorise(
      transformGateError * input.t());
  gradient.submat(transformWeight.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(transformGateError, 1);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void HighwayType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // If loading, delete the old layers and set size for weights.
  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];

    weights.set_size(inSize * inSize + inSize, 1);
  }

  ar(CEREAL_NVP(model));
  // ar(CEREAL_VECTOR_VARIANT_POINTER(network));
}

} // namespace ann
} // namespace mlpack

#endif
