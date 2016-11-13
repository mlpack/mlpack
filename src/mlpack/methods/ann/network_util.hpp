/**
 * @file network_util.hpp
 * @author Marcus Edel
 *
 * Neural network utilities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_NETWORK_UTIL_HPP
#define MLPACK_METHODS_ANN_NETWORK_UTIL_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer_traits.hpp>

/**
 * Neural network utility functions.
 */
namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Auxiliary function to get the number of weights of the specified network.
 *
 * @param network The network used for specifying the number of weights.
 * @return The number of weights.
 */
template<size_t I = 0, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), size_t>::type
NetworkSize(std::tuple<Tp...>& network);

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), size_t>::type
NetworkSize(std::tuple<Tp...>& network);

/**
 * Auxiliary function to get the number of weights of the specified layer.
 *
 * @param layer The layer used for specifying the number of weights.
 * @param output The layer output parameter.
 * @return The number of weights.
 */
template<typename T, typename P>
typename std::enable_if<
    !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerSize(T& layer, P& output);

template<typename T, typename P>
typename std::enable_if<
    HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerSize(T& layer, P& output);

/**
 * Auxiliary function to set the weights of the specified network.
 *
 * @param weights The weights used to set the weights of the network.
 * @param network The network used to set the weights.
 * @param offset The memory offset of the weights.
 */
template<size_t I = 0, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
NetworkWeights(arma::mat& weights,
               std::tuple<Tp...>& network,
               size_t offset = 0);

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
NetworkWeights(arma::mat& weights,
               std::tuple<Tp...>& network,
               size_t offset = 0);

/**
 * Auxiliary function to set the weights of the specified layer.
 *
 * @param layer The layer used to set the weights.
 * @param weights The weights used to set the weights of the layer.
 * @param offset The memory offset of the weights.
 * @param output The output parameter of the layer.
 * @return The number of weights.
 */
template<typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::mat&(T::*)()>::value, size_t>::type
LayerWeights(T& layer, arma::mat& weights, size_t offset, arma::mat& output);

template<typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::cube&(T::*)()>::value, size_t>::type
LayerWeights(T& layer, arma::mat& weights, size_t offset, arma::cube& output);

template<typename T, typename P>
typename std::enable_if<
    !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerWeights(T& layer, arma::mat& weights, size_t offset, P& output);

/**
 * Auxiliary function to set the gradients of the specified network.
 *
 * @param gradients The gradients used to set the gradient of the network.
 * @param network The network used to set the gradients.
 * @param offset The memory offset of the gradients.
 * return The number of gradients.
 */
template<size_t I = 0, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
NetworkGradients(arma::mat& gradients,
               std::tuple<Tp...>& network,
               size_t offset = 0);

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
NetworkGradients(arma::mat& gradients,
               std::tuple<Tp...>& network,
               size_t offset = 0);

/**
 * Auxiliary function to set the gradients of the specified layer.
 *
 * @param layer The layer used to set the gradients.
 * @param gradients The gradients used to set the gradient of the layer.
 * @param offset The memory offset of the gradients.
 * @param output The output parameter of the layer.
 * @return The number of gradients.
 */
template<typename T>
typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value, size_t>::type
LayerGradients(T& layer,
               arma::mat& gradients,
               size_t offset,
               arma::mat& output);

template<typename T>
typename std::enable_if<
    HasGradientCheck<T, arma::cube&(T::*)()>::value, size_t>::type
LayerGradients(T& layer,
               arma::mat& gradients,
               size_t offset,
               arma::cube& output);

template<typename T, typename P>
typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value, size_t>::type
LayerGradients(T& layer, arma::mat& gradients, size_t offset, P& output);

/**
 * Auxiliary function to get the input size of the specified network.
 *
 * @param network The network used for specifying the input size.
 * @return The input size.
 */
template<size_t I = 0, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), size_t>::type
NetworkInputSize(std::tuple<Tp...>& network);

template<size_t I, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), size_t>::type
NetworkInputSize(std::tuple<Tp...>& network);

/**
 * Auxiliary function to get the input size of the specified layer.
 *
 * @param layer The layer used for specifying the input size.
 * @param output The layer output parameter.
 * @return The input size.
 */
template<typename T, typename P>
typename std::enable_if<
    !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerInputSize(T& layer, P& output);

template<typename T, typename P>
typename std::enable_if<
    HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerInputSize(T& layer, P& output);

/**
 * Auxiliary function to set the weights of the specified network using a given
 * initialize rule.
 *
 * @param initializeRule The rule used to initialize the network weights.
 * @param weights The weights used to set the weights of the network.
 * @param network The network used to set the weights.
 * @param offset The memory offset of the weights.
 */
template<size_t I = 0, typename InitializationRuleType, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
NetworkWeights(InitializationRuleType& initializeRule,
               arma::mat& weights,
               std::tuple<Tp...>& network,
               size_t offset = 0);

template<size_t I, typename InitializationRuleType, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
NetworkWeights(InitializationRuleType& initializeRule,
               arma::mat& weights,
               std::tuple<Tp...>& network,
               size_t offset = 0);

/**
 * Auxiliary function to set the weights of the specified layer using the given
 * initialize rule.
 *
 * @param initializeRule The rule used to initialize the layer weights.
 * @param layer The layer used to set the weights.
 * @param weights The weights used to set the weights of the layer.
 * @param offset The memory offset of the weights.
 * @param output The output parameter of the layer.
 * @return The number of weights.
 */
template<typename InitializationRuleType, typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::mat&(T::*)()>::value, size_t>::type
LayerWeights(InitializationRuleType& initializeRule,
             T& layer,
             arma::mat& weights,
             size_t offset,
             arma::mat& output);

template<typename InitializationRuleType, typename T>
typename std::enable_if<
    HasWeightsCheck<T, arma::cube&(T::*)()>::value, size_t>::type
LayerWeights(InitializationRuleType& initializeRule,
             T& layer,
             arma::mat& weights,
             size_t offset,
             arma::cube& output);

template<typename InitializationRuleType, typename T, typename P>
typename std::enable_if<
    !HasWeightsCheck<T, P&(T::*)()>::value, size_t>::type
LayerWeights(InitializationRuleType& initializeRule,
             T& layer,
             arma::mat& weights,
             size_t offset,
             P& output);

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "network_util_impl.hpp"

#endif
