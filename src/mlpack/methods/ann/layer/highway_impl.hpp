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

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"
#include "../visitor/set_input_height_visitor.hpp"
#include "../visitor/set_input_width_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
Highway<InputDataType, OutputDataType, CustomLayers...>::Highway() :
    inSize(0),
    model(true),
    reset(false),
    width(0),
    height(0)
{
  // Nothing to do here.
}

template<
    typename InputDataType, typename OutputDataType, typename... CustomLayers>
Highway<InputDataType, OutputDataType, CustomLayers...>::Highway(
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

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
Highway<InputDataType, OutputDataType, CustomLayers...>::~Highway()
{
  if (!model)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      if (networkOwnerships[i])
        boost::apply_visitor(deleteVisitor, network[i]);
    }
  }
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void Highway<InputDataType, OutputDataType, CustomLayers...>::Reset()
{
  transformWeight = arma::mat(weights.memptr(), inSize, inSize, false, false);
  transformBias = arma::mat(weights.memptr() + transformWeight.n_elem,
      inSize, 1, false, false);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Highway<InputDataType, OutputDataType, CustomLayers...>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
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

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Highway<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  arma::Mat<eT> gyTransform = gy % transformGateActivation;
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, network.back()),
      gyTransform,
      boost::apply_visitor(deltaVisitor, network.back())),
      network.back());

  for (size_t i = 2; i < network.size() + 1; ++i)
  {
    boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
        outputParameterVisitor, network[network.size() - i]),
        boost::apply_visitor(deltaVisitor, network[network.size() - i + 1]),
        boost::apply_visitor(deltaVisitor,
        network[network.size() - i])), network[network.size() - i]);
  }

  g = boost::apply_visitor(deltaVisitor, network.front());

  transformGateError = gy % (networkOutput - inputParameter) %
      transformGateActivation % (1.0 - transformGateActivation);
  g += transformWeight.t() * transformGateError;
  g += gy % (1 - transformGateActivation);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Highway<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  arma::Mat<eT> errorTransform = error % transformGateActivation;
  boost::apply_visitor(GradientVisitor(boost::apply_visitor(
      outputParameterVisitor, network[network.size() - 2]),
      errorTransform), network.back());

  for (size_t i = 2; i < network.size(); ++i)
  {
    boost::apply_visitor(GradientVisitor(boost::apply_visitor(
        outputParameterVisitor, network[network.size() - i - 1]),
        boost::apply_visitor(deltaVisitor, network[network.size() - i + 1])),
        network[network.size() - i]);
  }

  boost::apply_visitor(GradientVisitor(input,
      boost::apply_visitor(deltaVisitor, network[1])), network.front());

  gradient.submat(0, 0, transformWeight.n_elem - 1, 0) = arma::vectorise(
      transformGateError * input.t());
  gradient.submat(transformWeight.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(transformGateError, 1);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void Highway<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  // If loading, delete the old layers and set size for weights.
  if (Archive::is_loading::value)
  {
    for (LayerTypes<CustomLayers...>& layer : network)
    {
      boost::apply_visitor(deleteVisitor, layer);
    }
    weights.set_size(inSize * inSize + inSize, 1);
  }

  ar & BOOST_SERIALIZATION_NVP(model);
  ar & BOOST_SERIALIZATION_NVP(network);
}

} // namespace ann
} // namespace mlpack

#endif
