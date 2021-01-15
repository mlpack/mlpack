/**
 * @file methods/ann/layer/sequential_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Sequential class, which acts as a feed-forward fully
 * connected network container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_IMPL_HPP

// In case it hasn't yet been included.
#include "sequential.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::
SequentialType(const bool model) :
    model(model), reset(false), width(0), height(0), ownsLayers(!model)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::
SequentialType(const bool model, const bool ownsLayers) :
    model(model), reset(false), width(0), height(0), ownsLayers(ownsLayers)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::
SequentialType(const SequentialType& layer) :
    model(layer.model),
    reset(layer.reset),
    width(layer.width),
    height(layer.height),
    ownsLayers(layer.ownsLayers)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>&
SequentialType<InputType, OutputType, Residual>::
operator = (const SequentialType& layer)
{
  if (this != &layer)
  {
    model = layer.model;
    reset = layer.reset;
    width = layer.width;
    height = layer.height;
    ownsLayers = layer.ownsLayers;
    parameters = layer.parameters;
    network.clear();
    // Build new layers according to source network.
    // for (size_t i = 0; i < layer.network.size(); ++i)
    // {
    //   this->network.push_back(boost::apply_visitor(copyVisitor,
    //       layer.network[i]));
    // }
  }
  return *this;
}


template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::~SequentialType()
{
  if (!model && ownsLayers)
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }
}

template <typename InputType, typename OutputType, bool Residual>
void SequentialType<InputType, OutputType, Residual>::
Forward(const InputType& input, OutputType& output)
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

  if (Residual)
  {
    if (arma::size(output) != arma::size(input))
    {
      Log::Fatal << "The sizes of the output and input matrices of the Residual"
          << " block should be equal. Please examine the network architecture."
          << std::endl;
    }
    output += input;
  }
}

template <typename InputType, typename OutputType, bool Residual>
void SequentialType<InputType, OutputType, Residual>::Backward(
        const InputType& /* input */,
        const OutputType& gy,
        OutputType& g)
{
  network.back()->Backward(
      network.back()->OutputParameter(),
      gy,
      network.back()->Delta()
  );

  for (size_t i = 2; i < network.size() + 1; ++i)
  {
    network[network.size() - i]->Backward(
        network[network.size() - i]->OutputParameter(),
        network[network.size() - i + 1]->Delta(),
        network[network.size() - i]->Delta()
    );
  }

  g = network.front()->Delta();

  if (Residual)
  {
    g += gy;
  }
}

template <typename InputType, typename OutputType, bool Residual>
void SequentialType<InputType, OutputType, Residual>::
Gradient(const InputType& input,
         const OutputType& error,
         OutputType& /* gradient */)
{
  network.back()->Gradient(
      network[network.size() - 2]->OutputParameter(),
      error,
      network.back()->Gradient()
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
      network.front()->Gradient()
  );
}

template <typename InputType, typename OutputType, bool Residual>
template<typename Archive>
void SequentialType<InputType, OutputType, Residual>::serialize(
        Archive& ar, const uint32_t /* version */)
{
  // If loading, delete the old layers.
  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }

  ar(CEREAL_NVP(model));
  // ar(CEREAL_VECTOR_VARIANT_POINTER(network));
  ar(CEREAL_NVP(ownsLayers));
}

} // namespace ann
} // namespace mlpack

#endif
