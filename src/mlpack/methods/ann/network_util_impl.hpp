/**
 * @file network_util_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the network auxiliary functions.
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

/**
 * Link the calculated activation with the connection layer.
 */
template<size_t I = 1, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
LinkParameter(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

template<size_t I = 1, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
LinkParameter(std::tuple<Tp...>& network)
{
  if (!LayerTraits<typename std::remove_reference<
      decltype(std::get<I>(network))>::type>::IsBiasLayer)
  {
    std::get<I>(network).InputParameter() = std::get<I - 1>(
        network).OutputParameter();
  }

  LinkParameter<I + 1, Tp...>(network);
}

/**
 * Run a single iteration of the feed forward algorithm, using the given
 * input and target vector, store the calculated error into the error
 * vector.
 */

template<size_t I = 0, typename DataType, typename... Tp>
void Forward(const DataType& input, std::tuple<Tp...>& network)
{
  std::get<I>(network).InputParameter() = input;

  std::get<I>(network).Forward(std::get<I>(network).InputParameter(),
                         std::get<I>(network).OutputParameter());

  ForwardTail<I + 1, Tp...>(network);
}

template<size_t I = 1, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type
ForwardTail(std::tuple<Tp...>& network)
{
  LinkParameter(network);
}

template<size_t I = 1, typename... Tp>
typename std::enable_if<I < sizeof...(Tp), void>::type
ForwardTail(std::tuple<Tp...>& network)
{
  std::get<I>(network).Forward(std::get<I - 1>(network).OutputParameter(),
      std::get<I>(network).OutputParameter());

  ForwardTail<I + 1, Tp...>(network);
}



/**
 * Run a single iteration of the feed backward algorithm, using the given
 * error of the output layer. Note that we iterate backward through the
 * layer modules.
 */
template<size_t I = 1, typename DataType, typename... Tp>
typename std::enable_if<I < (sizeof...(Tp) - 1), void>::type
Backward(const DataType& error, std::tuple<Tp...>& network)
{
  std::get<sizeof...(Tp) - I>(network).Backward(
      std::get<sizeof...(Tp) - I>(network).OutputParameter(), error,
      std::get<sizeof...(Tp) - I>(network).Delta());

  BackwardTail<I + 1, DataType, Tp...>(error, network);
}

template<size_t I = 1, typename DataType, typename... Tp>
typename std::enable_if<I == (sizeof...(Tp)), void>::type
BackwardTail(const DataType& /* unused */,
             std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

template<size_t I = 1, typename DataType, typename... Tp>
typename std::enable_if<I < (sizeof...(Tp)), void>::type
BackwardTail(const DataType& error, std::tuple<Tp...>& network)
{
  std::get<sizeof...(Tp) - I>(network).Backward(
      std::get<sizeof...(Tp) - I>(network).OutputParameter(),
      std::get<sizeof...(Tp) - I + 1>(network).Delta(),
      std::get<sizeof...(Tp) - I>(network).Delta());

  BackwardTail<I + 1, DataType, Tp...>(error, network);
}


/**
 * Iterate through all layer modules and update the the gradient using the
 * layer defined optimizer.
 */
template<
    typename LayerTypes,
    size_t I = 0,
    size_t Max = std::tuple_size<LayerTypes>::value - 1,
    typename... Tp
>
typename std::enable_if<I == Max, void>::type
UpdateGradients(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

template<
    typename LayerTypes,
    size_t I = 0,
    size_t Max = std::tuple_size<LayerTypes>::value - 1,
    typename... Tp
>
typename std::enable_if<I < Max, void>::type
UpdateGradients(std::tuple<Tp...>& network)
{
  Update(std::get<I>(network), std::get<I>(network).OutputParameter(),
         std::get<I + 1>(network).Delta());

  UpdateGradients<LayerTypes, I + 1, Max, Tp...>(network);
}

template<typename T, typename P, typename D>
typename std::enable_if<
    HasGradientCheck<T, P&(T::*)()>::value, void>::type
Update(T& layer, P& /* unused */, D& delta)
{
  layer.Gradient(layer.InputParameter(), delta, layer.Gradient());
}

template<typename T, typename P, typename D>
typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value, void>::type
Update(T& /* unused */, P& /* unused */, D& /* unused */)
{
  /* Nothing to do here */
}


template<typename eT>
void Pad(const arma::Mat<eT>& input, size_t wPad, size_t hPad, arma::Mat<eT>& output)
{
  if (output.n_rows != input.n_rows + wPad * 2 ||
      output.n_cols != input.n_cols + hPad * 2)
    output = arma::zeros(input.n_rows + wPad * 2, input.n_cols + hPad * 2);  
  output.submat(wPad, hPad, 
        wPad + input.n_rows - 1,
        hPad + input.n_cols - 1) = input;
}

template<typename eT>
void Pad(const arma::Cube<eT>& input, size_t wPad, size_t hPad, arma::Cube<eT>& output)
{
  output = arma::zeros(input.n_rows + wPad * 2, input.n_cols + hPad * 2, input.n_slices);
  for (size_t i = 0; i < input.n_slices; ++i)
    Pad<double>(input.slice(i), wPad, hPad, output.slice(i));
}


} // namespace ann
} // namespace mlpack

#endif

