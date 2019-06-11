/**
 * @file weight_norm_impl.hpp
 * @author Toshal Agrawal
 *
 * Implementation of the Weight Normalization Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_IMPL_HPP

// In case it is not included.
#include "weight_norm.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
WeightNorm<InputDataType, OutputDataType, CustomLayers...>::WeightNorm() :
    model(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
WeightNorm<InputDataType, OutputDataType, CustomLayers...>::~WeightNorm()
{
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Reset()
{
  size_t offset = boost::apply_visitor(WeightSetVisitor(std::move(weights),
      0), network[0]);
  boost::apply_visitor(resetVisitor, network[0]);

  vectorParameter = arma::mat(weights.memptr() + offset, offset, 1, false,
      false);

  scalarParameter = arma::mat(weights.memptr() + 2 * offset, 1, 1, false,
      false);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  weights.rows(0, networkWeightSize) = scalarParameter * vectorParameter /
      std::sqrt(arma::accu(arma::square(vectorParameter)));

  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, network[0]))),
      network[0]);

  output = boost::apply_visitor(outputParameterVisitor, network[0]);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, network[0])), std::move(gy), std::move(
      boost::apply_visitor(deltaVisitor, network[0]))), network[0]);

  g = boost::apply_visitor(deltaVisitor, network[0]);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  if (!model)
  {
    ResetGradients(gradient);
  }

  // Calculate the gradients of the wrapped layer.
  boost::apply_visitor(GradientVisitor(std::move(input),
      std::move(error)), network[0]);

  // Store the norm of vectorParameter temporarily.
  size_t normVectorParameter = std::sqrt(arma::accu(arma::square(
      vectorParameter)));

  // Calculate gradients of the scalar parameter.
  gradient[gradient.n_rows - 1] = arma::accu(gradient.rows(0, networkWeightSize)
      % vectorParameter) / normVectorParameter;

  // Calculate gradients of the vector parameter.
  gradient.rows(networkWeightSize, 2 * networkWeightSize) = scalarParameter /
      normVectorParameter * (gradient.rows(0, networkWeightSize) -
      gradient[gradient.n_rows - 1] / normVectorParameter * vectorParameter);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template <class LayerType, class... Args>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Add(
    Args... args)
{
  // Only one layer will be wrapped.
  if(network.size() > 0)
  {
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
    network.clear();
  }

  network.push_back(new LayerType(args...));

  // Now set the weights of the weight norm layer.
  networkWeightSize = boost::apply_visitor(weightSizeVisitor, network[0]);
  weights.set_size(2 * networkWeightSize + 1, 1);

  Reset();
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Add(
    LayerTypes<CustomLayers...> layer)
{
  // Only one layer will be wrapped.
  if(network.size() > 0)
  {
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
    network.clear();
  }

  network.push_back(layer);

  // Now set the weights of the weight norm layer.
  networkWeightSize = boost::apply_visitor(weightSizeVisitor, network[0]);
  weights.set_size(2 * networkWeightSize + 1, 1);

  Reset();
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::ResetGradients(
    arma::mat& gradient)
{
    boost::apply_visitor(GradientSetVisitor(std::move(gradient), 0),
        network[0]);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  if (Archive::is_loading::value)
  {
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
    network.clear();
  }

  ar & BOOST_SERIALIZATION_NVP(network);
  ar & BOOST_SERIALIZATION_NVP(model);
  ar & BOOST_SERIALIZATION_NVP(networkWeightSize);

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    // The behavior in earlier versions was to always assume the weights needed
    // to be reset.
    weights.set_size(2 * networkWeightSize + 1, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif
