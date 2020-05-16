/**
 * @file methods/ann/layer/multiply_merge_impl.hpp
 * @author Haritha Nair
 *
 * Definition of the MultiplyMerge module which multiplies the output of the
 * given modules element-wise.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_IMPL_HPP

// In case it hasn't yet been included.
#include "multiply_merge.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::MultiplyMerge(
    const bool model, const bool run) :
    model(model), run(run), ownsLayer(!model)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::~MultiplyMerge()
{
  if (ownsLayer)
  {
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
  }
}

template <typename InputDataType, typename OutputDataType,
          typename... CustomLayers>
template<typename InputType, typename OutputType>
void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::Forward(
    const InputType& input, OutputType& output)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(ForwardVisitor(input,
          boost::apply_visitor(outputParameterVisitor, network[i])),
          network[i]);
    }
  }

  output = boost::apply_visitor(outputParameterVisitor, network.front());
  for (size_t i = 1; i < network.size(); ++i)
  {
    output %= boost::apply_visitor(outputParameterVisitor, network[i]);
  }
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
          outputParameterVisitor, network[i]), gy,
          boost::apply_visitor(deltaVisitor, network[i])), network[i]);
    }

    g = boost::apply_visitor(deltaVisitor, network[0]);
    for (size_t i = 1; i < network.size(); ++i)
    {
      g += boost::apply_visitor(deltaVisitor, network[i]);
    }
  }
  else
    g = gy;
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& /* gradient */ )
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(GradientVisitor(input, error), network[i]);
    }
  }
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  // Be sure to clear other layers before loading.
  if (Archive::is_loading::value)
    network.clear();

  ar & BOOST_SERIALIZATION_NVP(network);
  ar & BOOST_SERIALIZATION_NVP(model);
  ar & BOOST_SERIALIZATION_NVP(run);
  ar & BOOST_SERIALIZATION_NVP(ownsLayer);
}

} // namespace ann
} // namespace mlpack

#endif
