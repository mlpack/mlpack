/**
 * @file ffn.hpp
 * @author Marcus Edel
 *
 * Definition of the FFN class, which implements feed forward neural networks.
 */
#ifndef __MLPACK_METHODS_ANN_FFN_HPP
#define __MLPACK_METHODS_ANN_FFN_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/network_traits.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard feed forward network.
 *
 * @tparam LayerTypes Contains all layer modules used to construct the network.
 * @tparam OutputLayerType The outputlayer type used to evaluate the network.
 * @tparam PerformanceFunction Performance strategy used to claculate the error.
 */
template <
  typename LayerTypes,
  typename OutputLayerType,
  class PerformanceFunction = CrossEntropyErrorFunction<>
>
class FFN
{
 public:
  /**
   * Construct the FFN object, which will construct a feed forward neural
   * network with the specified layers.
   *
   * @param network The network modules used to construct the network.
   * @param outputLayer The outputlayer used to evaluate the network.
   * @param performanceFunction Performance strategy used to claculate the error.
   */  
  template<typename Layer, typename OutLayer>
  FFN(Layer &&network, OutLayer &&outputLayer,
      PerformanceFunction performanceFunction = PerformanceFunction())
    : network(std::forward<Layer>(network)),
      outputLayer(std::forward<OutLayer>(outputLayer)),
      performanceFunc(std::move(performanceFunction)),
      trainError(0)
  {
    static_assert(std::is_same<typename std::decay<Layer>::type,
                  LayerTypes>::value,
                  "The type of network must be LayerTypes");
	static_assert(std::is_same<typename std::decay<OutLayer>::type,
                  OutputLayerType>::value,
                  "The type of outputLayer must be OutLayer");
  }  

  /**
   * Run a single iteration of the feed forward algorithm, using the given
   * input and target vector, store the calculated error into the error
   * parameter.
   *
   * @param input Input data used to evaluate the network.
   * @param target Target data used to calculate the network error.
   * @param error The calulated error of the output layer.
   */
  template <typename InputType, typename TargetType, typename ErrorType>
  void FeedForward(const InputType& input,
                   const TargetType& target,
                   ErrorType& error)
  {
    deterministic = false;
    trainError += Evaluate(input, target, error);
  }

  /**
   * Run a single iteration of the feed backward algorithm, using the given
   * error of the output layer.
   *
   * @param error The calulated error of the output layer.
   */
  template <typename InputType, typename ErrorType>
  void FeedBackward(const InputType& /* unused */, const ErrorType& error)
  {
    Backward<>(error, network);
    UpdateGradients<>(network);
  }

  /**
   * Update the weights using the layer defined optimizer.
   */
  void ApplyGradients()
  {
    ApplyGradients<>(network);

    // Reset the overall error.
    trainError = 0;
  }

  /**
   * Evaluate the network using the given input. The output activation is
   * stored into the output parameter.
   *
   * @param input Input data used to evaluate the network.
   * @param output Output data used to store the output activation
   */
  template <typename DataType>
  void Predict(const DataType& input, DataType& output)
  {
    deterministic = true;
    ResetParameter(network);

    Forward(input, network);
    OutputPrediction(output, network);
  }

  /**
   * Evaluate the trained network using the given input and compare the output
   * with the given target vector.
   *
   * @param input Input data used to evaluate the trained network.
   * @param target Target data used to calculate the network error.
   * @param error The calulated error of the output layer.
   */
  template <typename InputType, typename TargetType, typename ErrorType>
  double Evaluate(const InputType& input,
                  const TargetType& target,
                  ErrorType& error)
  {
    deterministic = false;
    ResetParameter(network);

    Forward(input, network);    
    return OutputError(target, error, network);
  }

  //! Get the error of the network.
  double Error() const
  {
    return trainError;
  }

  LayerTypes& Network()
  {
    return network;
  }

  LayerTypes const& Network() const
  {
    return network;
  }

 private:
  /**
   * Reset the network by zeroing the layer activations and by setting the
   * layer status.
   *
   * enable_if (SFINAE) is used to iterate through the network. The general
   * case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& t)
  {
    ResetDeterministic(std::get<I>(t));
    ResetParameter<I + 1, Tp...>(t);
  }

  /**
   * Reset the layer status by setting the current deterministic parameter
   * through all layer that implement the Deterministic function.
   *
   * enable_if (SFINAE) is used to iterate through the network. The general
   * case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<typename T>
  typename std::enable_if<
      HasDeterministicCheck<T, bool&(T::*)(void)>::value, void>::type
  ResetDeterministic(T& t)
  {
    t.Deterministic() = deterministic;
  }

  template<typename T>
  typename std::enable_if<
      !HasDeterministicCheck<T, bool&(T::*)(void)>::value, void>::type
  ResetDeterministic(T& /* unused */) { /* Nothing to do here */ }

  /**
   * Run a single iteration of the feed forward algorithm, using the given
   * input and target vector, store the calculated error into the error
   * vector.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * layer, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 0, typename DataType, typename... Tp>
  void Forward(const DataType& input, std::tuple<Tp...>& t)
  {
    std::get<I>(t).InputParameter() = input;

    std::get<I>(t).Forward(std::get<I>(t).InputParameter(),
                           std::get<I>(t).OutputParameter());
    
    ForwardTail<I + 1, Tp...>(t);
  }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& /* unused */)
  {
    LinkParameter(network);
  }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& t)
  {
    std::get<I>(t).Forward(std::get<I - 1>(t).OutputParameter(),
                           std::get<I>(t).OutputParameter());
    
    ForwardTail<I + 1, Tp...>(t);
  }

  /**
   * Link the calculated activation with the connection layer.
   *
   * enable_if (SFINAE) is used to iterate through the network. The general
   * case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LinkParameter(std::tuple<Tp ...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LinkParameter(std::tuple<Tp...>& t)
  {
    if (!LayerTraits<typename std::remove_reference<
        decltype(std::get<I>(t))>::type>::IsBiasLayer)
    {
      std::get<I>(t).InputParameter() = std::get<I - 1>(t).OutputParameter();
    }

    LinkParameter<I + 1, Tp...>(t);
  }

  /*
   * Calculate the output error and update the overall error.
   */
  template<typename DataType, typename ErrorType, typename... Tp>
  double OutputError(const DataType& target,
                     ErrorType& error,
                     const std::tuple<Tp...>& t)
  {
    // Calculate and store the output error.
    outputLayer.CalculateError(
        std::get<sizeof...(Tp) - 1>(t).OutputParameter(), target, error);

    // Measures the network's performance with the specified performance
    // function.
    return performanceFunc.Error(network, target, error);
  }

  /**
   * Run a single iteration of the feed backward algorithm, using the given
   * error of the output layer. Note that we iterate backward through the
   * layer modules.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * layer, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp) - 1), void>::type
  Backward(const DataType& error, std::tuple<Tp ...>& t)
  {
    std::get<sizeof...(Tp) - I>(t).Backward(
        std::get<sizeof...(Tp) - I>(t).OutputParameter(), error,
        std::get<sizeof...(Tp) - I>(t).Delta());    

    BackwardTail<I + 1, DataType, Tp...>(error, t);
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I == (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& /* unused */,
               std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& error, std::tuple<Tp...>& t)
  {    
    std::get<sizeof...(Tp) - I>(t).Backward(
        std::get<sizeof...(Tp) - I>(t).OutputParameter(),
        std::get<sizeof...(Tp) - I + 1>(t).Delta(),
        std::get<sizeof...(Tp) - I>(t).Delta());   

    BackwardTail<I + 1, DataType, Tp...>(error, t);
  }

  /**
   * Iterate through all layer modules and update the the gradient using the
   * layer defined optimizer.
   *
   * enable_if (SFINAE) is used to iterate through the network layer.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 1, typename... Tp>
  typename std::enable_if<I == Max, void>::type
  UpdateGradients(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 1, typename... Tp>
  typename std::enable_if<I < Max, void>::type
  UpdateGradients(std::tuple<Tp...>& t)
  {   
    Update(std::get<I>(t), std::get<I>(t).OutputParameter(),
           std::get<I + 1>(t).Delta());    
		   
    UpdateGradients<I + 1, Max, Tp...>(t);
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasGradientCheck<T, void(T::*)(const D&, P&)>::value, void>::type
  Update(T& t, P& /* unused */, D& delta)
  {
    t.Gradient(delta, t.Gradient());    
    t.Optimizer().Update();
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasGradientCheck<T, void(T::*)(const P&, D&)>::value, void>::type
  Update(T& /* unused */, P& /* unused */, D& /* unused */)
  {
    /* Nothing to do here */
  }

  /**
   * Update the weights using the calulated gradients.
   *
   * enable_if (SFINAE) is used to iterate through the network connections.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 1, typename... Tp>
  typename std::enable_if<I == Max, void>::type
  ApplyGradients(std::tuple<Tp...>& /* unused */)
  {
    /* Nothing to do here */
  }

  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 1, typename... Tp>
  typename std::enable_if<I < Max, void>::type
  ApplyGradients(std::tuple<Tp...>& t)
  {
    Apply(std::get<I>(t), std::get<I>(t).OutputParameter(),
          std::get<I + 1>(t).Delta());

    ApplyGradients<I + 1, Max, Tp...>(t);
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasGradientCheck<T, void(T::*)(const D&, P&)>::value, void>::type
  Apply(T& t, P& /* unused */, D& /* unused */)
  {
    t.Optimizer().Optimize();
    t.Optimizer().Reset();
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasGradientCheck<T, void(T::*)(const P&, D&)>::value, void>::type
  Apply(T& /* unused */, P& /* unused */, D& /* unused */)
  {
    /* Nothing to do here */
  }

  /*
   * Calculate and store the output activation.
   */
  template<typename DataType, typename... Tp>
  void OutputPrediction(DataType& output, std::tuple<Tp...>& t)
  {
    // Calculate and store the output prediction.
    outputLayer.OutputClass(std::get<sizeof...(Tp) - 1>(t).OutputParameter(),
        output);
  }

  //! The layer modules used to build the network.
  LayerTypes network;

  //! The outputlayer used to evaluate the network
  OutputLayerType& outputLayer;

  //! Performance strategy used to claculate the error.
  PerformanceFunction performanceFunc;

  //! The current training error of the network.
  double trainError;

  //! The current evaluation mode (training or testing).
  bool deterministic;
}; // class FFN

//! Network traits for the FFN network.
template <
  typename LayerTypes,
  typename OutputLayerType,
  class PerformanceFunction
>
class NetworkTraits<
    FFN<LayerTypes, OutputLayerType, PerformanceFunction> >
{
 public:
  static const bool IsFNN = true;
  static const bool IsRNN = false;
  static const bool IsCNN = false;
};

}; // namespace ann
}; // namespace mlpack

#endif
