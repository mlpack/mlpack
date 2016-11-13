/**
 * @file rnn.hpp
 * @author Marcus Edel
 *
 * Definition of the RNN class, which implements recurrent neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RNN_HPP
#define MLPACK_METHODS_ANN_RNN_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include <mlpack/methods/ann/network_util.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of a standard recurrent neural network.
 *
 * @tparam LayerTypes Contains all layer modules used to construct the network.
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam PerformanceFunction Performance strategy used to calculate the error.
 */
template <
  typename LayerTypes,
  typename OutputLayerType,
  typename InitializationRuleType = NguyenWidrowInitialization,
  class PerformanceFunction = CrossEntropyErrorFunction<>
>
class RNN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = RNN<LayerTypes,
                          OutputLayerType,
                          InitializationRuleType,
                          PerformanceFunction>;

  /**
   * Create the RNN object with the given predictors and responses set (this is
   * the set that is used to train the network) and the given optimizer.
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * @param network Network modules used to construct the network.
   * @param outputLayer Output layer used to evaluate the network.
   * @param predictors Input training variables.
   * @param responses Outputs resulting from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   * @param performanceFunction Optional instantiated PerformanceFunction
   *        object used to calculate the error.
   */
  template<typename LayerType,
           typename OutputType,
           template<typename> class OptimizerType>
  RNN(LayerType &&network,
      OutputType &&outputLayer,
      const arma::mat& predictors,
      const arma::mat& responses,
      OptimizerType<NetworkType>& optimizer,
      InitializationRuleType initializeRule = InitializationRuleType(),
      PerformanceFunction performanceFunction = PerformanceFunction());

  /**
   * Create the RNN object with the given predictors and responses set (this is
   * the set that is used to train the network). Optionally, specify which
   * initialize rule and performance function should be used.
   *
   * @param network Network modules used to construct the network.
   * @param outputLayer Output layer used to evaluate the network.
   * @param predictors Input training variables.
   * @param responses Outputs resulting from input training variables.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   * @param performanceFunction Optional instantiated PerformanceFunction
   *        object used to calculate the error.
   */
  template<typename LayerType, typename OutputType>
  RNN(LayerType &&network,
      OutputType &&outputLayer,
      const arma::mat& predictors,
      const arma::mat& responses,
      InitializationRuleType initializeRule = InitializationRuleType(),
      PerformanceFunction performanceFunction = PerformanceFunction());

  /**
   * Create the RNN object with an empty predictors and responses set and
   * default optimizer. Make sure to call Train(predictors, responses) when
   * training.
   *
   * @param network Network modules used to construct the network.
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   * @param performanceFunction Optional instantiated PerformanceFunction
   *        object used to calculate the error.
   */
  template<typename LayerType, typename OutputType>
  RNN(LayerType &&network,
      OutputType &&outputLayer,
      InitializationRuleType initializeRule = InitializationRuleType(),
      PerformanceFunction performanceFunction = PerformanceFunction());

  /**
   * Train the recurrent neural network on the given input data. By default, the
   * SGD optimization algorithm is used, but others can be specified
   * (such as mlpack::optimization::RMSprop).
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
      template<typename> class OptimizerType = mlpack::optimization::SGD
  >
  void Train(const arma::mat& predictors, const arma::mat& responses);

  /**
   * Train the recurrent neural network with the given instantiated optimizer.
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
      template<typename> class OptimizerType = mlpack::optimization::SGD
  >
  void Train(OptimizerType<NetworkType>& optimizer);

  /**
   * Train the recurrent neural network on the given input data using the given
   * optimizer.
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
      template<typename> class OptimizerType = mlpack::optimization::SGD
  >
  void Train(const arma::mat& predictors,
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
  void Predict(arma::mat& predictors, arma::mat& responses);

  /**
   * Evaluate the recurrent neural network with the given parameters. This
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
   * Evaluate the gradient of the recurrent neural network with the given
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

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  /*
   * Predict the response of the given input matrix.
   */
  template <typename DataType>
  void SinglePredict(const DataType& input, DataType& output)
  {
    deterministic = true;
    seqLen = input.n_rows / inputSize;
    ResetParameter(network);

    // Iterate through the input sequence and perform the feed forward pass.
    for (seqNum = 0; seqNum < seqLen; seqNum++)
    {
      // Perform the forward pass and save the activations.
      Forward(input.rows(seqNum * inputSize, (seqNum + 1) * inputSize - 1),
          network);
      SaveActivations(network);

      // Retrieve output of the subsequence.
      if (seqOutput)
      {
        DataType seqOutput;
        OutputPrediction(seqOutput, network);
        output = arma::join_cols(output, seqOutput);
      }
    }

    // Retrieve output of the complete sequence.
    if (!seqOutput)
      OutputPrediction(output, network);
  }

  /**
   * Reset the network by clearing the layer activations and by setting the
   * layer status.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& /* unused */)
  {
    activations.clear();
  }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& network)
  {
    ResetDeterministic(std::get<I>(network));
    ResetSeqLen(std::get<I>(network));
    ResetRecurrent(std::get<I>(network), std::get<I>(network).InputParameter());
    std::get<I>(network).Delta().zeros();

    ResetParameter<I + 1, Tp...>(network);
  }

  /**
   * Reset the layer status by setting the current deterministic parameter
   * for all layer that implement the Deterministic function.
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
  ResetDeterministic(T& /* unused */) { /* Nothing to do here */ }

  /**
   * Reset the layer sequence length by setting the current seqLen parameter
   * for all layer that implement the SeqLen function.
   */
  template<typename T>
  typename std::enable_if<
      HasSeqLenCheck<T, size_t&(T::*)(void)>::value, void>::type
  ResetSeqLen(T& layer)
  {
    layer.SeqLen() = seqLen;
  }

  template<typename T>
  typename std::enable_if<
      !HasSeqLenCheck<T, size_t&(T::*)(void)>::value, void>::type
  ResetSeqLen(T& /* unused */) { /* Nothing to do here */ }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when resetting
   * the recurrent parameter.
   */
  template<typename T, typename P>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  ResetRecurrent(T& layer, P& /* unused */)
  {
    layer.RecurrentParameter().zeros();
  }

  template<typename T, typename P>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  ResetRecurrent(T& /* unused */, P& /* unused */)
  {
    /* Nothing to do here */
  }

  /**
   * Initialize the network by setting the input size and output size.
   */
  template<size_t I = 0, typename InputDataType, typename TargetDataType,
      typename... Tp>
  typename std::enable_if<I == sizeof...(Tp) - 1, void>::type
  InitLayer(const InputDataType& /* unused */,
            const TargetDataType& target,
            std::tuple<Tp...>& /* unused */)
  {
    seqOutput = outputSize < target.n_elem ? true : false;
  }

  template<size_t I = 0, typename InputDataType, typename TargetDataType,
      typename... Tp>
  typename std::enable_if<I < sizeof...(Tp) - 1, void>::type
  InitLayer(const InputDataType& input,
            const TargetDataType& target,
            std::tuple<Tp...>& network)
  {
    Init(std::get<I>(network), std::get<I>(network).OutputParameter(),
       std::get<I + 1>(network).Delta());

    InitLayer<I + 1, InputDataType, TargetDataType, Tp...>(input, target,
        network);
  }

  /**
   * Retrieve the weight matrix for all layer that implement the Weights
   * function to extract the input size and output size.
   */
  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasGradientCheck<T, P&(T::*)()>::value, void>::type
  Init(T& layer, P& /* unused */, D& /* unused */)
  {
    // Initialize the input size only once.
    if (!inputSize)
      inputSize = layer.Weights().n_cols;

    outputSize = layer.Weights().n_rows;
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value, void>::type
  Init(T& /* unused */, P& /* unused */, D& /* unused */)
  {
    /* Nothing to do here */
  }

  /**
   * Save the network layer activations.
   */
  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I == Max, void>::type
  SaveActivations(std::tuple<Tp...>& /* unused */)
  {
    Save(I, std::get<I>(network), std::get<I>(network).InputParameter());
    LinkRecurrent(network);
  }

  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I < Max, void>::type
  SaveActivations(std::tuple<Tp...>& network)
  {
    Save(I, std::get<I>(network), std::get<I>(network).InputParameter());
    SaveActivations<I + 1, Max, Tp...>(network);
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when storing
   * the activations.
   */
  template<typename T, typename P>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Save(const size_t layerNumber, T& layer, P& /* unused */)
  {
    if (activations.size() == layerNumber)
    {
      activations.push_back(new arma::mat(layer.RecurrentParameter().n_rows,
          seqLen));
    }

    activations[layerNumber].unsafe_col(seqNum) = layer.RecurrentParameter();
  }

  template<typename T, typename P>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Save(const size_t layerNumber, T& layer, P& /* unused */)
  {
    if (activations.size() == layerNumber)
    {
      activations.push_back(new arma::mat(layer.OutputParameter().n_rows,
          seqLen));
    }

    activations[layerNumber].unsafe_col(seqNum) = layer.OutputParameter();
  }

  /**
   * Load the network layer activations.
   */
  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename DataType, typename... Tp
  >
  typename std::enable_if<I == Max, void>::type
  LoadActivations(DataType& input, std::tuple<Tp...>& network)
  {
    Load(I, std::get<I>(network), std::get<I>(network).InputParameter());
    std::get<0>(network).InputParameter() = input;
  }

  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename DataType, typename... Tp
  >
  typename std::enable_if<I < Max, void>::type
  LoadActivations(DataType& input, std::tuple<Tp...>& network)
  {
    Load(I, std::get<I>(network), std::get<I>(network).InputParameter());
    LoadActivations<I + 1, Max, DataType, Tp...>(input, network);
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when storing
   * the activations.
   */
  template<typename T, typename P>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Load(const size_t layerNumber, T& layer, P& /* unused */)
  {
    layer.RecurrentParameter() = activations[layerNumber].unsafe_col(seqNum);
  }

  template<typename T, typename P>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Load(const size_t layerNumber, T& layer, P& /* unused */)
  {
    layer.OutputParameter() = activations[layerNumber].unsafe_col(seqNum);
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
  ForwardTail(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& network)
  {
    std::get<I>(network).Forward(std::get<I - 1>(network).OutputParameter(),
        std::get<I>(network).OutputParameter());

    ForwardTail<I + 1, Tp...>(network);
  }

  /**
   * Link the calculated activation with the correct layer.
   */
  template<
      size_t I = 1,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I == Max, void>::type
  LinkParameter(std::tuple<Tp ...>& /* unused */)
  {
    if (!LayerTraits<typename std::remove_reference<
        decltype(std::get<I>(network))>::type>::IsBiasLayer)
    {
      std::get<I>(network).InputParameter() = std::get<I - 1>(
          network).OutputParameter();
    }
  }

  template<
      size_t I = 1,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I < Max, void>::type
  LinkParameter(std::tuple<Tp...>& network)
  {
    if (!LayerTraits<typename std::remove_reference<
        decltype(std::get<I>(network))>::type>::IsBiasLayer)
    {
      std::get<I>(network).InputParameter() = std::get<I - 1>(
          network).OutputParameter();
    }

    LinkParameter<I + 1, Max, Tp...>(network);
  }

  /**
   * Link the calculated activation with the correct recurrent layer.
   */
  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I == Max, void>::type
  LinkRecurrent(std::tuple<Tp ...>& /* unused */) { /* Nothing to do here */ }

  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I < Max, void>::type
  LinkRecurrent(std::tuple<Tp...>& network)
  {
    UpdateRecurrent(std::get<I>(network), std::get<I>(network).InputParameter(),
        std::get<I + 1>(network).OutputParameter());
    LinkRecurrent<I + 1, Max, Tp...>(network);
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when updating
   * the recurrent activations.
   */
  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  UpdateRecurrent(T& layer, P& /* unused */, D& output)
  {
    layer.RecurrentParameter() = output;
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  UpdateRecurrent(T& /* unused */, P& /* unused */, D& /* unused */)
  {
    /* Nothing to do here */
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
  void Backward(DataType& error, std::tuple<Tp ...>& network)
  {
    std::get<sizeof...(Tp) - I>(network).Backward(
        std::get<sizeof...(Tp) - I>(network).OutputParameter(), error,
        std::get<sizeof...(Tp) - I>(network).Delta());

    BackwardTail<I + 1, DataType, Tp...>(error, network);
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I == (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& /* unused */, std::tuple<Tp...>& /* unused */)
  {
    /* Nothing to do here */
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& error, std::tuple<Tp...>& network)
  {
    BackwardRecurrent(std::get<sizeof...(Tp) - I - 1>(network),
        std::get<sizeof...(Tp) - I - 1>(network).InputParameter(),
        std::get<sizeof...(Tp) - I + 1>(network).Delta());

    std::get<sizeof...(Tp) - I>(network).Backward(
        std::get<sizeof...(Tp) - I>(network).OutputParameter(),
        std::get<sizeof...(Tp) - I + 1>(network).Delta(),
        std::get<sizeof...(Tp) - I>(network).Delta());

    BackwardTail<I + 1, DataType, Tp...>(error, network);
  }

  /*
   * Update the delta of the recurrent layer.
   */
  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  BackwardRecurrent(T& layer, P& /* unused */, D& delta)
  {
    if (!layer.Delta().is_empty())
      delta += layer.Delta();
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  BackwardRecurrent(T& /* unused */, P& /* unused */, D& /* unused */)
  {
    /* Nothing to do here */
  }

  /**
   * Iterate through all layer modules and update the the gradient using the
   * layer defined optimizer.
   */
  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 2,
      typename... Tp>
  typename std::enable_if<I == Max, void>::type
  UpdateGradients(std::tuple<Tp...>& network)
  {
    Update(std::get<I>(network), std::get<I>(network).OutputParameter(),
        std::get<I + 1>(network).Delta(), std::get<I + 1>(network),
        std::get<I + 1>(network).InputParameter(),
        std::get<I + 1>(network).Delta());
  }

  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 2,
      typename... Tp>
  typename std::enable_if<I < Max, void>::type
  UpdateGradients(std::tuple<Tp...>& network)
  {
    Update(std::get<I>(network), std::get<I>(network).OutputParameter(),
        std::get<I + 1>(network).Delta(), std::get<I + 1>(network),
        std::get<I + 1>(network).InputParameter(),
        std::get<I + 2>(network).Delta());

    UpdateGradients<I + 1, Max, Tp...>(network);
  }

  template<typename T1, typename P1, typename D1, typename T2, typename P2,
      typename D2>
  typename std::enable_if<
      HasGradientCheck<T1, P1&(T1::*)()>::value &&
      HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value, void>::type
  Update(T1& layer, P1& /* unused */, D1& /* unused */, T2& /* unused */,
         P2& /* unused */, D2& delta2)
  {
    layer.Gradient(layer.InputParameter(), delta2, layer.Gradient());
  }

  template<typename T1, typename P1, typename D1, typename T2, typename P2,
      typename D2>
  typename std::enable_if<
      (!HasGradientCheck<T1, P1&(T1::*)()>::value &&
      !HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value) ||
      (!HasGradientCheck<T1, P1&(T1::*)()>::value &&
      HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value), void>::type
  Update(T1& /* unused */, P1& /* unused */, D1& /* unused */, T2& /* unused */,
         P2& /* unused */, D2& /* unused */)
  {
    /* Nothing to do here */
  }

  template<typename T1, typename P1, typename D1, typename T2, typename P2,
      typename D2>
  typename std::enable_if<
      HasGradientCheck<T1, P1&(T1::*)()>::value &&
      !HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value, void>::type
  Update(T1& layer, P1& /* unused */, D1& delta1, T2& /* unused */,
         P2& /* unused */, D2& /* unused */)
  {
    layer.Gradient(layer.InputParameter(), delta1, layer.Gradient());
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

  //! Instantiated recurrent neural network.
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
  arma::mat predictors;

  //! The matrix of responses to the input data points.
  arma::mat responses;

  //! Locally stored network input size.
  size_t inputSize;

  //! Locally stored network output size.
  size_t outputSize;

  //! The index of the current sequence number.
  size_t seqNum;

  //! Locally stored number of samples in one input sequence.
  size_t seqLen;

  //! Locally stored parameter that indicates if the input is a sequence.
  bool seqOutput;

  //! The activation storage we are using to perform the feed backward pass.
  boost::ptr_vector<arma::mat> activations;

  //! The number of separable functions (the number of predictor points).
  size_t numFunctions;

  //! Locally stored backward error.
  arma::mat error;
}; // class RNN

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "rnn_impl.hpp"

#endif
