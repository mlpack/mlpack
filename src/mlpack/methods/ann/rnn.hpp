/**
 * @file rnn.hpp
 * @author Marcus Edel
 *
 * Definition of the RNN class, which implements feed forward neural networks.
 */
#ifndef __MLPACK_METHODS_ANN_RNN_HPP
#define __MLPACK_METHODS_ANN_RNN_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp> 

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
  class PerformanceFunction = CrossEntropyErrorFunction<>,
  typename MatType = arma::mat
>
class RNN
{
 public:
  /**
   * Construct the RNN object, which will construct a recurrent neural
   * network with the specified layers.
   *
   * @param network The network modules used to construct the network.
   * @param outputLayer The outputlayer used to evaluate the network.
   * @param performanceFunction Performance strategy used to claculate the error.
   */
  RNN(const LayerTypes& network, OutputLayerType& outputLayer,
      PerformanceFunction performanceFunction = PerformanceFunction()) :
      network(network),
      outputLayer(outputLayer),
      performanceFunction(std::move(performanceFunction)),
      trainError(0),
      inputSize(0),
      outputSize(0)
  {
    // Nothing to do here.
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
  void FeedBackward(const InputType& input, const ErrorType& error)
  {
    // Iterate through the input sequence and perform the feed backward pass.
    for (seqNum = seqLen - 1; seqNum >= 0; seqNum--)
    {
      // Load the network activation for the upcoming backward pass.
        LoadActivations(input.rows(seqNum * inputSize, (seqNum + 1) *
            inputSize - 1), network);

      // Perform the backward pass.
      if (seqOutput)
      {
        ErrorType seqError = error.unsafe_col(seqNum);
        Backward(seqError, network);
      }
      else
      {
        Backward(error, network);
      }
      
      // Link the parameters and update the gradients.
      LinkParameter(network);
      UpdateGradients<>(network);

      if (seqNum == 0) break;
    }
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
   * Evaluate the network using the given input and compare the output with the
   * given target vector.
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
    // Initialize the activation storage only once.
    if (activations.empty())
      InitLayer(input, target, network);

    double networkError = 0;
    seqLen = input.n_rows / inputSize;
    deterministic = false;
    ResetParameter(network);

    error = ErrorType(outputSize, outputSize < target.n_elem ? seqLen : 1);

    // Iterate through the input sequence and perform the feed forward pass.
    for (seqNum = 0; seqNum < seqLen; seqNum++)
    {
      // Perform the forward pass and save the activations.
      Forward(input.rows(seqNum * inputSize, (seqNum + 1) * inputSize - 1),
          network);
      SaveActivations(network);

      // Retrieve output error of the subsequence.
      if (seqOutput)
      {
        arma::mat seqError = error.unsafe_col(seqNum);
        arma::mat seqTarget = target.submat(seqNum * outputSize, 0,
            (seqNum + 1) * outputSize - 1, 0);
        networkError += OutputError(seqTarget, seqError, network);
      }
    }

    // Retrieve output error of the complete sequence.
    if (!seqOutput)
      return OutputError(target, error, network);

    return networkError;
  }

  //! Get the error of the network.
  double Error() const
  {
    return trainError;
  }

 private:
  /**
   * Reset the network by clearing the layer activations and by setting the
   * layer status.
   *
   * enable_if (SFINAE) is used to iterate through the network. The general
   * case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& /* unused */)
  {
    activations.clear();
  }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& t)
  {
    ResetDeterministic(std::get<I>(t));
    ResetSeqLen(std::get<I>(t));
    ResetRecurrent(std::get<I>(t), std::get<I>(t).InputParameter());
    std::get<I>(t).Delta().zeros();

    ResetParameter<I + 1, Tp...>(t);
  }

  /**
   * Reset the layer status by setting the current deterministic parameter
   * for all layer that implement the Deterministic function.
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
   * Reset the layer sequence length by setting the current seqLen parameter
   * for all layer that implement the SeqLen function.
   */
  template<typename T>
  typename std::enable_if<
      HasSeqLenCheck<T, size_t&(T::*)(void)>::value, void>::type
  ResetSeqLen(T& t)
  {
    t.SeqLen() = seqLen;
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
  ResetRecurrent(T& t, P& /* unused */)
  {
    t.RecurrentParameter().zeros();
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
   *
   * enable_if (SFINAE) is used to iterate through the network. The general
   * case peels off the first type and recurses, as usual with
   * variadic function templates.
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
            std::tuple<Tp...>& t)
  {
    Init(std::get<I>(t), std::get<I>(t).OutputParameter(),
       std::get<I + 1>(t).Delta());
    
    InitLayer<I + 1, InputDataType, TargetDataType, Tp...>(input, target, t);
  }

  /**
   * Retrieve the weight matrix for all layer that implement the Weights
   * function to extract the input size and output size.
   */
  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasGradientCheck<T, void(T::*)(const D&, P&)>::value, void>::type
  Init(T& t, P& /* unused */, D& /* unused */)
  {
    // Initialize the input size only once.
    if (!inputSize)
      inputSize = t.Weights().n_cols;

    outputSize = t.Weights().n_rows;
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasGradientCheck<T, void(T::*)(const P&, D&)>::value, void>::type
  Init(T& /* unused */, P& /* unused */, D& /* unused */)
  {
    /* Nothing to do here */
  }

  /**
   * Save the network layer activations.
   *
   * enable_if (SFINAE) is used to iterate through the network layer.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  SaveActivations(std::tuple<Tp...>& /* unused */)
  {
    LinkRecurrent(network);
  }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  SaveActivations(std::tuple<Tp...>& t)
  {
    Save(I, std::get<I>(t), std::get<I>(t).InputParameter());
    SaveActivations<I + 1, Tp...>(t);
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when storing
   * the activations.
   */
  template<typename T, typename P>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Save(const size_t layerNumber, T& t, P& /* unused */)
  {
    if (activations.size() == layerNumber)
      activations.push_back(new MatType(t.RecurrentParameter().n_rows, seqLen));

    activations[layerNumber].unsafe_col(seqNum) = t.RecurrentParameter();
  }

  template<typename T, typename P>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Save(const size_t layerNumber, T& t, P& /* unused */)
  {
    if (activations.size() == layerNumber)
      activations.push_back(new MatType(t.OutputParameter().n_rows, seqLen));

    activations[layerNumber].unsafe_col(seqNum) = t.OutputParameter();
  }

  /**
   * Load the network layer activations.
   *
   * enable_if (SFINAE) is used to iterate through the network layer.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LoadActivations(DataType& input, std::tuple<Tp...>& t)
  {
    std::get<0>(t).InputParameter() = input;
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LoadActivations(DataType& input, std::tuple<Tp...>& t)
  {
    Load(I, std::get<I>(t), std::get<I>(t).InputParameter());
    LoadActivations<I + 1, DataType, Tp...>(input, t);
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when storing
   * the activations.
   */
  template<typename T, typename P>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Load(const size_t layerNumber, T& t, P& /* unused */)
  {
    t.RecurrentParameter() = activations[layerNumber].unsafe_col(seqNum);
  }

  template<typename T, typename P>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Load(const size_t layerNumber, T& t, P& /* unused */)
  {
    t.OutputParameter() = activations[layerNumber].unsafe_col(seqNum);
  }

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
  ForwardTail(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& t)
  {
    std::get<I>(t).Forward(std::get<I - 1>(t).OutputParameter(),
        std::get<I>(t).OutputParameter());

    ForwardTail<I + 1, Tp...>(t);
  }

  /**
   * Link the calculated activation with the correct layer.
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

  /**
   * Link the calculated activation with the correct recurrent layer.
   *
   * enable_if (SFINAE) is used to iterate through the network. The general
   * case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == (sizeof...(Tp) - 1), void>::type
  LinkRecurrent(std::tuple<Tp ...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp) - 1), void>::type
  LinkRecurrent(std::tuple<Tp...>& t)
  {
    UpdateRecurrent(std::get<I>(t), std::get<I>(t).InputParameter(),
        std::get<I + 1>(t).OutputParameter());
    LinkRecurrent<I + 1, Tp...>(t);
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when updating
   * the recurrent activations.
   */
  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  UpdateRecurrent(T& t, P& /* unused */, D& output)
  {
    t.RecurrentParameter() = output;
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
                     const std::tuple<Tp...>& t)
  {
    // Calculate and store the output error.
    outputLayer.CalculateError(
        std::get<sizeof...(Tp) - 1>(t).OutputParameter(), target, error);

    // Masures the network's performance with the specified performance
    // function.
    return performanceFunction.Error(network, target, error);
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
  BackwardTail(const DataType& /* unused */, std::tuple<Tp...>& /* unused */)
  {
    /* Nothing to do here */
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& error, std::tuple<Tp...>& t)
  {
    BackwardRecurrent(std::get<sizeof...(Tp) - I - 1>(t),
        std::get<sizeof...(Tp) - I - 1>(t).InputParameter(),
        std::get<sizeof...(Tp) - I + 1>(t).Delta());
    
    std::get<sizeof...(Tp) - I>(t).Backward(
        std::get<sizeof...(Tp) - I>(t).OutputParameter(),
        std::get<sizeof...(Tp) - I + 1>(t).Delta(),
        std::get<sizeof...(Tp) - I>(t).Delta());

    BackwardTail<I + 1, DataType, Tp...>(error, t);
  }

  /*
   * Update the delta of the recurrent layer.
   */
  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  BackwardRecurrent(T& t, P& /* unused */, D& delta)
  {
    if (!t.Delta().is_empty())
      delta += t.Delta();
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
   *
   * enable_if (SFINAE) is used to iterate through the network layer.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 2,
      typename... Tp>
  typename std::enable_if<I == Max, void>::type
  UpdateGradients(std::tuple<Tp...>& t)
  {
    Update(std::get<I>(t), std::get<I>(t).OutputParameter(),
        std::get<I + 1>(t).Delta(), std::get<I + 1>(t),
        std::get<I + 1>(t).InputParameter(), std::get<I + 1>(t).Delta());
  }

  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 2,
      typename... Tp>
  typename std::enable_if<I < Max, void>::type
  UpdateGradients(std::tuple<Tp...>& t)
  {
    Update(std::get<I>(t), std::get<I>(t).OutputParameter(),
        std::get<I + 1>(t).Delta(), std::get<I + 1>(t),
        std::get<I + 1>(t).InputParameter(), std::get<I + 2>(t).Delta());

    UpdateGradients<I + 1, Max, Tp...>(t);
  }

  template<typename T1, typename P1, typename D1, typename T2, typename P2,
      typename D2>
  typename std::enable_if<
      HasGradientCheck<T1, void(T1::*)(const D1&, P1&)>::value &&
      HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value, void>::type
  Update(T1& t1, P1& /* unused */, D1& /* unused */, T2& /* unused */,
         P2& /* unused */, D2& delta2)
  {
    t1.Gradient(delta2, t1.Gradient());
    t1.Optimizer().Update();
  }

  template<typename T1, typename P1, typename D1, typename T2, typename P2,
      typename D2>
  typename std::enable_if<
      (!HasGradientCheck<T1, void(T1::*)(const D1&, P1&)>::value &&
      !HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value) ||
      (!HasGradientCheck<T1, void(T1::*)(const D1&, P1&)>::value &&
      HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value), void>::type
  Update(T1& /* unused */, P1& /* unused */, D1& /* unused */, T2& /* unused */,
         P2& /* unused */, D2& /* unused */)
  {
    /* Nothing to do here */
  }

  template<typename T1, typename P1, typename D1, typename T2, typename P2,
      typename D2>
  typename std::enable_if<
      HasGradientCheck<T1, void(T1::*)(const D1&, P1&)>::value &&
      !HasRecurrentParameterCheck<T2, P2&(T2::*)()>::value, void>::type
  Update(T1& t1, P1& /* unused */, D1& delta1, T2& /* unused */,
         P2& /* unused */, D2& /* unused */)
  {
    t1.Gradient(delta1, t1.Gradient());
    t1.Optimizer().Update();
  }

  /**
   * Update the weights using the calulated gradients.
   *
   * enable_if (SFINAE) is used to iterate through the network layer.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp>
  typename std::enable_if<I == Max, void>::type
  ApplyGradients(std::tuple<Tp...>& /* unused */)
  {
    /* Nothing to do here */
  }

  template<size_t I = 0, size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp>
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
  PerformanceFunction performanceFunction;

  //! The current training error of the network.
  double trainError;

  //! Locally stored network input size.
  size_t inputSize;

  //! Locally stored network output size.
  size_t outputSize;

  //! The current evaluation mode (training or testing).
  bool deterministic;

  //! The index of the current sequence number.
  size_t seqNum;

  //! Locally stored number of samples in one input sequence.
  size_t seqLen;

  //! Locally stored parameter that indicates if the input is a sequence.
  bool seqOutput;

  //! The activation storage we are using to perform the feed backward pass.
  boost::ptr_vector<MatType> activations;
}; // class RNN

//! Network traits for the RNN network.
template <
  typename LayerTypes,
  typename OutputLayerType,
  class PerformanceFunction
>
class NetworkTraits<
    RNN<LayerTypes, OutputLayerType, PerformanceFunction> >
{
 public:
  static const bool IsFNN = false;
  static const bool IsRNN = true;
  static const bool IsCNN = false;
  static const bool IsSAE = false;
};

} // namespace ann
} // namespace mlpack

#endif
