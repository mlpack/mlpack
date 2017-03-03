/**
 * @file sequential_impl.hpp
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

template <typename InputDataType, typename OutputDataType>
Sequential<InputDataType, OutputDataType>::Sequential(
    const bool model) : model(model), reset(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
Sequential<InputDataType, OutputDataType>::~Sequential()
{
  if (!model)
  {
    for (LayerTypes& layer : network)
    {
      boost::apply_visitor(deleteVisitor, layer);
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sequential<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, network.front()))),
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
      boost::apply_visitor(SetInputWidthVisitor(width, true), network[i]);

      // Set the input height.
      boost::apply_visitor(SetInputHeightVisitor(height, true), network[i]);
    }

    boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[i - 1])), std::move(
        boost::apply_visitor(outputParameterVisitor, network[i]))),
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
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sequential<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, network.back())), std::move(gy),
      std::move(boost::apply_visitor(deltaVisitor, network.back()))),
      network.back());

  for (size_t i = 2; i < network.size() + 1; ++i)
  {
    boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[network.size() - i])), std::move(
        boost::apply_visitor(deltaVisitor, network[network.size() - i + 1])),
        std::move(boost::apply_visitor(deltaVisitor,
        network[network.size() - i]))), network[network.size() - i]);
  }

  g = boost::apply_visitor(deltaVisitor, network.front());
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sequential<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(error)),
      network.front());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[i - 1])), std::move(
        boost::apply_visitor(deltaVisitor, network[i + 1]))), network[i]);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Sequential<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */, const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
