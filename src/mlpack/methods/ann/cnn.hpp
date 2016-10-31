/**
 * @file cnn.hpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Definition of the CNN class, which implements convolutional neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CNN_HPP
#define MLPACK_METHODS_ANN_CNN_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/network_util.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard convolutional network.
 *
 * @tparam LayerTypes Contains all layer modules used to construct the network.
 * @tparam OutputLayerType The outputlayer type used to evaluate the network.
 * @tparam PerformanceFunction Performance strategy used to calculate the error.
 */
template <
  typename LayerTypes,
  typename OutputLayerType,
  typename InitializationRuleType = NguyenWidrowInitialization,
  class PerformanceFunction = CrossEntropyErrorFunction<>
>
class CNN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = CNN<LayerTypes,
                          OutputLayerType,
                          InitializationRuleType,
                          PerformanceFunction>;

  /**
   * Create the CNN object with the given predictors and responses set (this is
   * the set that is used to train the network) and the given optimizer.
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * @param network Network modules used to construct the network.
   * @param outputLayer Outputlayer used to evaluate the network.
   * @param predictors Input training variables.
   * @param responses Outputs resulting from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network paramter.
   * @param performanceFunction Optional instantiated PerformanceFunction
   *        object used to claculate the error.
   */
  template<typename LayerType,
           typename OutputType,
           template<typename> class OptimizerType>
  CNN(LayerType &&network,
      OutputType &&outputLayer,
      const arma::cube& predictors,
      const arma::mat& responses,
      OptimizerType<NetworkType>& optimizer,
      InitializationRuleType initializeRule = InitializationRuleType(),
      PerformanceFunction performanceFunction = PerformanceFunction());

  /**
   * Create the CNN object with the given predictors and responses set (this is
   * the set that is used to train the network). Optionally, specify which
   * initialize rule and performance function should be used.
   *
   * @param network Network modules used to construct the network.
   * @param outputLayer Outputlayer used to evaluate the network.
   * @param predictors Input training variables.
   * @param responses Outputs resulting from input training variables.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network paramter.
   * @param performanceFunction Optional instantiated PerformanceFunction
   *        object used to claculate the error.
   */
  template<typename LayerType, typename OutputType>
  CNN(LayerType &&network,
      OutputType &&outputLayer,
      const arma::cube& predictors,
      const arma::mat& responses,
      InitializationRuleType initializeRule = InitializationRuleType(),
      PerformanceFunction performanceFunction = PerformanceFunction());

  /**
   * Create the CNN object with an empty predictors and responses set and
   * default optimizer. Make sure to call Train(predictors, responses) when
   * training.
   *
   * @param network Network modules used to construct the network.
   * @param outputLayer Outputlayer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network paramter.
   * @param performanceFunction Optional instantiated PerformanceFunction
   *        object used to claculate the error.
   */
  template<typename LayerType, typename OutputType>
  CNN(LayerType &&network,
      OutputType &&outputLayer,
      InitializationRuleType initializeRule = InitializationRuleType(),
      PerformanceFunction performanceFunction = PerformanceFunction());
  /**
   * Train the convolutional neural network on the given input data. By default, the
   * RMSprop optimization algorithm is used, but others can be specified
   * (such as mlpack::optimization::SGD).
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   */
  template<
      template<typename> class OptimizerType = mlpack::optimization::RMSprop
  >
  void Train(const arma::cube& predictors, const arma::mat& responses);

  /**
   * Train the convolutional neural network with the given instantiated optimizer.
   * Using this overload allows configuring the instantiated optimizer before
   * training is performed.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @param optimizer Instantiated optimizer used to train the model.
   */
  template<
      template<typename> class OptimizerType = mlpack::optimization::RMSprop
  >
  void Train(OptimizerType<NetworkType>& optimizer);

  /**
   * Train the convolutional neural network on the given input data using the
   * given optimizer.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   */
  template<
      template<typename> class OptimizerType = mlpack::optimization::RMSprop
  >
  void Train(const arma::cube& predictors,
             const arma::mat& responses,
             OptimizerType<NetworkType>& optimizer);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * OutputClass() function.
   *
   * @param predictors Input predictors.
   * @param responses Matrix to put output predictions of responses into.
   */
  void Predict(arma::cube& predictors, arma::mat& responses);

  /**
   * Evaluate the convolutional neural network with the given parameters. This
   * function is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   * @param i Index of point to use for objective function evaluation.
   * @param deterministic Whether or not to train or test the model. Note some
   * layer act differently in training or testing mode.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const bool deterministic = true);

  /**
   * Evaluate the gradient of the convolutional neural network with the given
   * parameters, and with respect to only one point in the dataset. This is
   * useful for optimizers such as SGD, which require a separable objective
   * function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param i Index of points to use for objective function gradient evaluation.
   * @param gradient Matrix to output gradient into.
   */
  void Gradient(const arma::mat& parameters,
                const size_t i,
                arma::mat& gradient);

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  /**
   * Serialize the convolutional neural network.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Reset the network by setting the layer status.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& network)
  {
    ResetDeterministic(std::get<I>(network));
    ResetParameter<I + 1, Tp...>(network);
  }

  /**
   * Reset the layer status by setting the current deterministic parameter
   * through all layer that implement the Deterministic function.
   */
  template<typename T>
  typename std::enable_if<
      HasDeterministicCheck<T, bool&(T::*)(void)>::value, void>::type
  ResetDeterministic(T& layer)
  {
    layer.Deterministic() = deterministic;
  }

  template<typename T>
  typename std::enable_if<
      !HasDeterministicCheck<T, bool&(T::*)(void)>::value, void>::type
  ResetDeterministic(T& /* unused */) { /* Nothing to do here */
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

  /*
   * Calculate the output error and update the overall error.
   */
  template<typename DataType, typename ErrorType, typename... Tp>
  double OutputError(const DataType& target,
                     ErrorType& error,
                     const std::tuple<Tp...>& network)
  {
    // Calculate and store the output error.
    outputLayer.CalculateError(
        std::get<sizeof...(Tp) - 1>(network).OutputParameter(), target, error);

    // Masures the network's performance with the specified performance
    // function.
    return performanceFunc.Error(network, target, error);
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
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I == Max, void>::type
  UpdateGradients(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I < Max, void>::type
  UpdateGradients(std::tuple<Tp...>& network)
  {
    Update(std::get<I>(network), std::get<I>(network).OutputParameter(),
           std::get<I + 1>(network).Delta());

    UpdateGradients<I + 1, Max, Tp...>(network);
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

  /*
   * Calculate and store the output activation.
   */
  template<typename DataType, typename... Tp>
  void OutputPrediction(DataType& output, std::tuple<Tp...>& network)
  {
    // Calculate and store the output prediction.
    outputLayer.OutputClass(std::get<sizeof...(Tp) - 1>(
        network).OutputParameter(), output);
  }

  //! Instantiated convolutional neural network.
  LayerTypes network;

  //! The outputlayer used to evaluate the network
  OutputLayerType& outputLayer;

  //! Performance strategy used to claculate the error.
  PerformanceFunction performanceFunc;

  //! The current evaluation mode (training or testing).
  bool deterministic;

  //! Matrix of (trained) parameters.
  arma::mat parameter;

  //! The matrix of data points (predictors).
  arma::cube predictors;

  //! The matrix of responses to the input data points.
  arma::mat responses;

  //! The number of separable functions (the number of predictor points).
  size_t numFunctions;

  //! Locally stored backward error.
  arma::mat error;

  //! Locally stored sample size.
  size_t sampleSize;
}; // class CNN

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "cnn_impl.hpp"

#endif
