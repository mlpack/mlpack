#ifndef MLPACK_METHODS_ANN_ANN_IMPL_HPP
#define MLPACK_METHODS_ANN_ANN_IMPL_HPP

// In case it hasn't been included yet.
#include "ann.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
double ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::Forward(const arma::mat& input)
{
  return forward_call<size_t, std::integral_constant<size_t, 0>>::call(input, layers, this);
}

/** For forward pass */
template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename T, size_t LayerNum>
double ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::forward_call<T, std::integral_constant<size_t, LayerNum>>::call(const arma::mat& input, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network)
{
  std::get<LayerNum>(layers).Forward(std::move(input), std::move(std::get<LayerNum>(layers).OutputParameter()));
  return forward_call<T, std::integral_constant<size_t, LayerNum + 1>>::call(std::get<LayerNum>(layers).OutputParameter(), layers, network);
}
  
template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename T>
double ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::forward_call<T, std::integral_constant<size_t, NumLayers>>::call(const arma::mat& input, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network)
{
  return std::get<NumLayers>(layers).Forward(std::move(input), std::move(network->CurrentTarget()));
}
/** forward pass finish */

/** Reset pass */
template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename T, size_t LayerNum>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::reset_call_layer<T, std::integral_constant<size_t, LayerNum>>::call(InitializationRuleType& initRule, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network)
{
  reset_call<decltype(std::get<LayerNum>(layers))>::call(initRule, std::get<LayerNum>(layers));
  reset_call_layer<size_t, std::integral_constant<size_t, LayerNum + 1>>::call(initRule, layers, network);
}
  
template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename T>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::reset_call_layer<T, std::integral_constant<size_t, NumLayers>>::call(InitializationRuleType& initRule, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network)
{
  reset_call<decltype(std::get<NumLayers>(layers))>::call(initRule, std::get<NumLayers>(layers));
}
/** reset pass finish */

/** for backward pass */
template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename UpdatePolicy>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::backward_call<UpdatePolicy, std::integral_constant<size_t, NumLayers>>::call(UpdatePolicy& policy, arma::mat& error, arma::mat& currentTarget, arma::mat& currentInput, std::tuple<LayerTypes..., OutputLayerType>& layers)
{
  std::cout << NumLayers << " Start" << std::endl;
  std::get<NumLayers>(layers).BackwardAndGradient(std::get<NumLayers - 1>(layers).OutputParameter(), currentTarget, policy, error);
  std::cout << NumLayers << " End" << std::endl;
  backward_call<UpdatePolicy, std::integral_constant<size_t, NumLayers - 1>>::call(policy, error, currentTarget, currentInput, layers);
}

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename UpdatePolicy, size_t LayerNum>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::backward_call<UpdatePolicy, std::integral_constant<size_t, LayerNum>>::call(UpdatePolicy& policy, arma::mat& error, arma::mat& currentTarget, arma::mat& currentInput, std::tuple<LayerTypes..., OutputLayerType>& layers)
{
  arma::mat tempError;
  std::cout << LayerNum << " Start" << std::endl;
  std::get<LayerNum>(layers).BackwardAndGradient(std::get<LayerNum - 1>(layers).OutputParameter(), error, policy, tempError);
  std::cout << LayerNum << " Start" << std::endl;
  error = tempError;
  backward_call<UpdatePolicy, std::integral_constant<size_t, LayerNum - 1>>::call(policy, error, currentTarget, currentInput, layers);
}

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename UpdatePolicy>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::backward_call<UpdatePolicy, std::integral_constant<size_t, 0>>::call(UpdatePolicy& policy, arma::mat& error, arma::mat& currentTarget, arma::mat& currentInput, std::tuple<LayerTypes..., OutputLayerType>& layers)
{
  arma::mat tempError;
  std::cout << 0 << " Start" << std::endl;
  std::get<0>(layers).BackwardAndGradient(currentInput, error, policy, tempError);
  std::cout << 0 << " End" << std::endl;
}

/** backward pass finish */

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<
    template<typename, typename...> class OptimizerType,
    typename... OptimizerTypeArgs
>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::Train(
      const arma::mat& predictors,
      const arma::mat& responses,
      OptimizerType<NetworkType, OptimizerTypeArgs...>& optimizer)
{
  numFunctions = responses.n_cols;
  
  ResetParameters();

  ResetData(predictors, responses);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
double ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::Evaluate(const arma::mat& /* parameters */, 
                                         const size_t i, const bool deterministic)
{
  //if (deterministic != this->deterministic)
  //{
  //  this->deterministic = deterministic;
  //  ResetDeterministic();
  //}

  currentInput = predictors.unsafe_col(i);
  currentTarget = responses.unsafe_col(i);

  return Forward(std::move(currentInput));
}

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::ResetData(const arma::mat &predictors,
                                          const arma::mat &responses)
{
  numFunctions = responses.n_cols;
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);
}

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::ResetParameters()
{
  auto initRule = InitializationRuleType();
  reset_call_layer<size_t, std::integral_constant<size_t, 0>>::call(initRule, layers, this);
}

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::Gradient(
    const arma::mat& parameters, const size_t i, arma::mat& gradient)
{
  std::cout << Evaluate(parameters, i, false) << std::endl;
}

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
template<typename UpdatePolicyType, typename... Params>
void ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...>::UpdateGradient(UpdatePolicyType& updatePolicy, Params... params)
{
  backward_call<UpdatePolicyType, std::integral_constant<size_t, NumLayers>>::call(updatePolicy, error, currentTarget, currentInput, layers);
}

} // namespace ann
} // namespace mlpack

#endif
