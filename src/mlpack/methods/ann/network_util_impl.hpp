/**
 * @file network_util_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the network auxiliary functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_NETWORK_UTIL_IMPL_HPP
#define MLPACK_METHODS_ANN_NETWORK_UTIL_IMPL_HPP

#include "network_util_impl.hpp"

#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann {

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), size_t>::type
NetworkSize(std::tuple<Tp...>& /* unused */)
{
  return 0;
}

template<size_t I, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), size_t>::type
NetworkSize(std::tuple<Tp...>& network)
{
  return LayerSize(std::get<I>(network), std::get<I>(
      network).OutputParameter()) + NetworkSize<I + 1, Tp...>(network);
}

template<typename T, typename P>
typename std::enable_if<
  HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerSize(T& layer, P& /* unused */)
{
  return layer.Weights().n_elem;
}

template<typename T, typename P>
typename std::enable_if<
  !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerSize(T& /* unused */, P& /* unused */)
{
  return 0;
}

template<size_t I, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
NetworkWeights(arma::mat& weights,
               std::tuple<Tp...>& network,
               size_t offset)
{
  NetworkWeights<I + 1, Tp...>(weights, network,
      offset + LayerWeights(std::get<I>(network), weights,
      offset, std::get<I>(network).OutputParameter()));

}

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
NetworkWeights(arma::mat& /* unused */,
               std::tuple<Tp...>& /* unused */,
               size_t /* unused */)
{
  /* Nothing to do here */
}

template<typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::mat&(T::*)()>::value, size_t>::type
LayerWeights(T& layer,
             arma::mat& weights,
             size_t offset,
             arma::mat& /* unused */)
{
  layer.Weights() = arma::mat(weights.memptr() + offset,
      layer.Weights().n_rows, layer.Weights().n_cols, false, false);

  return layer.Weights().n_elem;
}

template<typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::cube&(T::*)()>::value, size_t>::type
LayerWeights(T& layer,
             arma::mat& weights,
             size_t offset,
             arma::cube& /* unused */)
{
  layer.Weights() = arma::cube(weights.memptr() + offset,
      layer.Weights().n_rows, layer.Weights().n_cols,
      layer.Weights().n_slices, false, false);

  return layer.Weights().n_elem;
}

template<typename T, typename P>
typename std::enable_if<
    !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerWeights(T& /* unused */,
             arma::mat& /* unused */,
             size_t /* unused */,
             P& /* unused */)
{
  return 0;
}

template<size_t I, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
NetworkGradients(arma::mat& gradients,
                 std::tuple<Tp...>& network,
                 size_t offset)
{
  NetworkGradients<I + 1, Tp...>(gradients, network,
      offset + LayerGradients(std::get<I>(network), gradients,
      offset, std::get<I>(network).OutputParameter()));
}

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
NetworkGradients(arma::mat& /* unused */,
               std::tuple<Tp...>& /* unused */,
               size_t /* unused */)
{
  /* Nothing to do here */
}

template<typename T>
typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value, size_t>::type
LayerGradients(T& layer,
               arma::mat& gradients,
               size_t offset,
               arma::mat& /* unused */)
{
  layer.Gradient() = arma::mat(gradients.memptr() + offset,
      layer.Weights().n_rows, layer.Weights().n_cols, false, false);

  return layer.Weights().n_elem;
}

template<typename T>
typename std::enable_if<
    HasGradientCheck<T, arma::cube&(T::*)()>::value, size_t>::type
LayerGradients(T& layer,
               arma::mat& gradients,
               size_t offset,
               arma::cube& /* unused */)
{
  layer.Gradient() = arma::cube(gradients.memptr() + offset,
      layer.Weights().n_rows, layer.Weights().n_cols,
      layer.Weights().n_slices, false, false);

  return layer.Weights().n_elem;
}

template<typename T, typename P>
typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value, size_t>::type
LayerGradients(T& /* unused */,
               arma::mat& /* unused */,
               size_t /* unused */,
               P& /* unused */)
{
  return 0;
}

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), size_t>::type
NetworkInputSize(std::tuple<Tp...>& /* unused */)
{
  return 0;
}

template<size_t I, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), size_t>::type
NetworkInputSize(std::tuple<Tp...>& network)
{
  const size_t inputSize = LayerInputSize(std::get<I>(network), std::get<I>(
      network).OutputParameter());

  if (inputSize)
  {
    return inputSize;
  }

  return NetworkInputSize<I + 1, Tp...>(network);
}

template<typename T, typename P>
typename std::enable_if<
  HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerInputSize(T& layer, P& /* unused */)
{
  return layer.Weights().n_cols;
}

template<typename T, typename P>
typename std::enable_if<
  !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerInputSize(T& /* unused */, P& /* unused */)
{
  return 0;
}

template<size_t I, typename InitializationRuleType, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
NetworkWeights(InitializationRuleType& initializeRule,
               arma::mat& weights,
               std::tuple<Tp...>& network,
               size_t offset)
{
  NetworkWeights<I + 1, InitializationRuleType, Tp...>(initializeRule, weights,
      network, offset + LayerWeights(initializeRule, std::get<I>(network),
      weights, offset, std::get<I>(network).OutputParameter()));
}

template<size_t I, typename InitializationRuleType, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
NetworkWeights(InitializationRuleType& /* initializeRule */,
               arma::mat& /* weights */,
               std::tuple<Tp...>& /* network */,
               size_t /* offset */)
{
  /* Nothing to do here */
}

template<typename InitializationRuleType, typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::mat&(T::*)()>::value, size_t>::type
LayerWeights(InitializationRuleType& initializeRule,
             T& layer,
             arma::mat& weights,
             size_t offset,
             arma::mat& /* output */)
{
  layer.Weights() = arma::mat(weights.memptr() + offset,
      layer.Weights().n_rows, layer.Weights().n_cols, false, false);

  initializeRule.Initialize(layer.Weights(), layer.Weights().n_rows,
      layer.Weights().n_cols);

  return layer.Weights().n_elem;
}

template<typename InitializationRuleType, typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::cube&(T::*)()>::value, size_t>::type
LayerWeights(InitializationRuleType& initializeRule,
             T& layer,
             arma::mat& weights,
             size_t offset,
             arma::cube& /* output */)
{
  layer.Weights() = arma::cube(weights.memptr() + offset,
      layer.Weights().n_rows, layer.Weights().n_cols,
      layer.Weights().n_slices, false, false);

  initializeRule.Initialize(layer.Weights(), layer.Weights().n_rows,
      layer.Weights().n_cols);

  return layer.Weights().n_elem;
}

template<typename InitializationRuleType, typename T, typename P>
typename std::enable_if<
    !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerWeights(InitializationRuleType& /* initializeRule */,
             T& /* layer */,
             arma::mat& /* weights */,
             size_t /* offset */,
             P& /* output */)
{
  return 0;
}

} // namespace ann
} // namespace mlpack

#endif
