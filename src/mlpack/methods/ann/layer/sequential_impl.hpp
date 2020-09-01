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

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"
#include "../visitor/set_input_height_visitor.hpp"
#include "../visitor/set_input_width_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType, bool Residual,
          typename... CustomLayers>
Sequential<InputDataType, OutputDataType, Residual, CustomLayers...>::
Sequential(const bool model) :
    model(model), reset(false), width(0), height(0), ownsLayers(!model)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType, bool Residual,
          typename... CustomLayers>
Sequential<InputDataType, OutputDataType, Residual, CustomLayers...>::
Sequential(const bool model, const bool ownsLayers) :
    model(model), reset(false), width(0), height(0), ownsLayers(ownsLayers)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType, bool Residual,
          typename... CustomLayers>
Sequential<InputDataType, OutputDataType, Residual, CustomLayers...>::
Sequential(const Sequential& layer) :
    model(layer.model),
    reset(layer.reset),
    width(layer.width),
    height(layer.height),
    ownsLayers(layer.ownsLayers)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType, bool Residual,
          typename... CustomLayers>
Sequential<InputDataType, OutputDataType, Residual, CustomLayers...>&
Sequential<InputDataType, OutputDataType, Residual, CustomLayers...>::
operator = (const Sequential& layer)
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
    for (size_t i = 0; i < layer.network.size(); ++i)
    {
      this->network.push_back(boost::apply_visitor(copyVisitor,
          layer.network[i]));
    }
  }
  return *this;
}


template <typename InputDataType, typename OutputDataType, bool Residual,
          typename... CustomLayers>
Sequential<
    InputDataType, OutputDataType, Residual, CustomLayers...>::~Sequential()
{
  if (!model && ownsLayers)
  {
    for (LayerTypes<CustomLayers...>& layer : network)
      boost::apply_visitor(deleteVisitor, layer);
  }
}

template<typename InputDataType, typename OutputDataType, bool Residual,
         typename... CustomLayers>
template<typename eT>
void Sequential<InputDataType, OutputDataType, Residual, CustomLayers...>::
Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  boost::apply_visitor(ForwardVisitor(input,
      boost::apply_visitor(outputParameterVisitor, network.front())),
      network.front());

  if (!reset)
  {
    if (boost::apply_visitor(outputWidthVisitor, network.front()) != 0)
    {
      width = boost::apply_visitor(outputWidthVisitor, network.front());
    }

    if (boost::apply_visitor(outputHeightVisitor, network.front()) != 0)
    {
      height = boost::apply_visitor(outputHeightVisitor, network.front());
    }
  }

  for (size_t i = 1; i < network.size(); ++i)
  {
    if (!reset)
    {
      // Set the input width.
      boost::apply_visitor(SetInputWidthVisitor(width), network[i]);

      // Set the input height.
      boost::apply_visitor(SetInputHeightVisitor(height), network[i]);
    }

    boost::apply_visitor(ForwardVisitor(boost::apply_visitor(
        outputParameterVisitor, network[i - 1]),
        boost::apply_visitor(outputParameterVisitor, network[i])),
        network[i]);

    if (!reset)
    {
      // Get the output width.
      if (boost::apply_visitor(outputWidthVisitor, network[i]) != 0)
      {
        width = boost::apply_visitor(outputWidthVisitor, network[i]);
      }

      // Get the output height.
      if (boost::apply_visitor(outputHeightVisitor, network[i]) != 0)
      {
        height = boost::apply_visitor(outputHeightVisitor, network[i]);
      }
    }
  }

  if (!reset)
  {
    reset = true;
  }

  output = boost::apply_visitor(outputParameterVisitor, network.back());

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

template<typename InputDataType, typename OutputDataType, bool Residual,
         typename... CustomLayers>
template<typename eT>
void Sequential<
    InputDataType, OutputDataType, Residual, CustomLayers...>::Backward(
        const arma::Mat<eT>& /* input */,
        const arma::Mat<eT>& gy,
        arma::Mat<eT>& g)
{
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, network.back()), gy,
      boost::apply_visitor(deltaVisitor, network.back())),
      network.back());

  for (size_t i = 2; i < network.size() + 1; ++i)
  {
    boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
        outputParameterVisitor, network[network.size() - i]),
        boost::apply_visitor(deltaVisitor, network[network.size() - i + 1]),
        boost::apply_visitor(deltaVisitor, network[network.size() - i])),
        network[network.size() - i]);
  }

  g = boost::apply_visitor(deltaVisitor, network.front());

  if (Residual)
  {
    g += gy;
  }
}

template<typename InputDataType, typename OutputDataType, bool Residual,
         typename... CustomLayers>
template<typename eT>
void Sequential<InputDataType, OutputDataType, Residual, CustomLayers...>::
Gradient(const arma::Mat<eT>& input,
         const arma::Mat<eT>& error,
         arma::Mat<eT>& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(boost::apply_visitor(
      outputParameterVisitor, network[network.size() - 2]), error),
      network.back());

  for (size_t i = 2; i < network.size(); ++i)
  {
    boost::apply_visitor(GradientVisitor(boost::apply_visitor(
        outputParameterVisitor, network[network.size() - i - 1]),
        boost::apply_visitor(deltaVisitor, network[network.size() - i + 1])),
        network[network.size() - i]);
  }

  boost::apply_visitor(GradientVisitor(input,
      boost::apply_visitor(deltaVisitor, network[1])), network.front());
}

template<typename InputDataType, typename OutputDataType, bool Residual,
         typename... CustomLayers>
template<typename Archive>
void Sequential<
    InputDataType, OutputDataType, Residual, CustomLayers...>::serialize(
        Archive& ar, const unsigned int version)
{
  // If loading, delete the old layers.
  if (Archive::is_loading::value)
  {
    for (LayerTypes<CustomLayers...>& layer : network)
    {
      boost::apply_visitor(deleteVisitor, layer);
    }
  }

  ar & BOOST_SERIALIZATION_NVP(model);
  ar & BOOST_SERIALIZATION_NVP(network);

  if (version >= 1)
    ar & BOOST_SERIALIZATION_NVP(ownsLayers);
  else if (Archive::is_loading::value)
    ownsLayers = !model;
}

} // namespace ann
} // namespace mlpack

#endif
