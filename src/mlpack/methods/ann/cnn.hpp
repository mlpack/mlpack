/**
 * @file cnn.hpp
 * @author Shangtong Zhang
 *
 * Definition of the CNN class, which implements convolutional neural networks.
 */
#ifndef __MLPACK_METHODS_ANN_CNN_HPP
#define __MLPACK_METHODS_ANN_CNN_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include <mlpack/methods/ann/network_traits.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard convolutional network.
 *
 * @tparam ConnectionTypes Tuple that contains all layer module and 
 * connection module which will be used to construct the network.
 * These tuples should be organized as 
 * <layer, connection, layer, ....., connection, layer>
 * @tparam OutputLayerType The outputlayer type used to evaluate the network.
 * @tparam PerformanceFunction Performance strategy used to claculate the error.
 * @tparam MaType Type of the gradients. (arma::mat or arma::sp_mat).
 */
template <
    typename ConnectionTypes,
    typename OutputLayerType,
    class PerformanceFunction = CrossEntropyErrorFunction<>,
    typename MatType = arma::mat
>
class CNN
{
 public:
  /**
   * Construct the CNN object, which will construct a convolutional neural
   * network with the specified layers.
   *
   * @param network The network modules used to construct net network.
   * @param outputLayer The outputlayer used to evaluate the network.
   */
  CNN(const ConnectionTypes& network, OutputLayerType& outputLayer)
      : network(network), outputLayer(outputLayer), trainError(0), seqNum(0)
  {
    // Nothing to do here.
  }

  /**
   * Run a single iteration of the feed forward algorithm, using the given
   * input and target vector, updating the resulting error into the error
   * vector.
   *
   * @param input Input data used to evaluate the network.
   * @param target Target data used to calculate the network error.
   * @param error The calulated error of the output layer.
   * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
   */
  template <typename VecType>
  void FeedForward(const MatType& input,
                   const VecType& target,
                   VecType& error)
  {
    seqNum++;
    trainError += Evaluate(input, target, error);
  }
  
  /**
   * Reset all connection module and layer module in the network.
   */
  void Reset() {
    ResetLayer(network);
    ResetConnection(network);
  }

  /**
   * Run a single iteration of the feed backward algorithm, using the given
   * error of the output layer.
   *
   * @param error The calulated error of the output layer.
   * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
   */
  template <typename VecType>
  void FeedBackward(const VecType& error)
  {
    // Initialize the gradient storage only once.
    if (!gradients.size())
      InitLayer(network);
    
    gradientNum = 0;
    FeedBackward(network, error);
    UpdateGradients(network);
  }

  /**
   * Updating the weights using the specified optimizer.
   */
  void ApplyGradients()
  {
    gradientNum = 0;
    ApplyGradients(network);

    // Reset the overall error.
    trainError = 0;
    seqNum = 0;
  }

  /**
   * Evaluate the network using the given input. The output activation is
   * stored into the output parameter.
   *
   * @param input Input data used to evaluate the network.
   * @param output Output data used to store the output activation
   * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
   */
  template <typename VecType>
  void Predict(const MatType& input, VecType& output)
  {
    Reset();

    std::get<0>(std::get<0>(network)).InputActivation() = input;

    FeedForward(network);
    OutputPrediction(network, output);
  }
  
  /**
   * Evaluate the trained network using the given input and compare the output
   * with the given target vector.
   *
   * @param input Input data used to evaluate the trained network.
   * @param target Target data used to calculate the network error.
   * @param error The calulated error of the output layer.
   * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
   */
  template <typename VecType>
  double Evaluate(const MatType& input, const VecType& target, VecType& error)
  {
    Reset();
    std::get<0>(std::get<0>(network)).InputActivation() = input;
    FeedForward(network);
    return OutputError(network, target, error);
  }

  //! Get the error of the network.
  double Error() const { return trainError; }

 private:
  /**
   * Helper function to reset all layer module
   * by zeroing the layer activations
   * and delta which store the passed error in backward propagation.
   *
   * enable_if (SFINAE) is used to iterate through the network layer
   * modules. The general case peels off the first type and recurses, as usual
   * with variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I >= sizeof...(Tp), void>::type
  ResetLayer(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetLayer(std::tuple<Tp...>& t)
  {
    ResetL(std::get<I>(t));
    ResetLayer<I + 2, Tp...>(t);
  }
  
  
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetL(std::tuple<Tp...>& /* unused */) { }
  
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetL(std::tuple<Tp...>& t)
  {
    std::get<I>(t).InputActivation().zeros();
    std::get<I>(t).Delta().zeros();
    ResetL<I + 1, Tp...>(t);
  }
  
  /**
   * Helper function to reset all connection module
   * by zeroing the connection weight delta
   * and delta which store the passed error in backward propagation.
   *
   * enable_if (SFINAE) is used to iterate through the network connections.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I >= sizeof...(Tp), void>::type
  ResetConnection(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetConnection(std::tuple<Tp...>& t)
  {
    ResetC(std::get<I>(t));
    ResetConnection<I + 2, Tp...>(t);
  }
  
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetC(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetC(std::tuple<Tp...>& t)
  {
    std::get<I>(t).Delta().zeros();
    std::get<I>(t).WeightsDelta().zeros();
    ResetC<I + 1, Tp...>(t);
  }

  /**
   * Run a single iteration of the feed forward algorithm, using the given
   * input and target vector, updating the resulting error into the error
   * vector.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * connections, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I >= sizeof...(Tp), void>::type
  FeedForward(std::tuple<Tp...>& /* unused */) { }
  
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  FeedForward(std::tuple<Tp...>& t)
  {
    ConnectionForward(std::get<I>(t));
    
    LayerForward(std::get<I + 1>(t));

    FeedForward<I + 2, Tp...>(t);
  }

  /**
   * Sum up all connection activations by evaluating all connections.
   *
   * enable_if (SFINAE) is used to iterate through the network connections.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ConnectionForward(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ConnectionForward(std::tuple<Tp...>& t)
  {
    std::get<I>(t).FeedForward(std::get<I>(t).InputLayer().InputActivation());
    ConnectionForward<I + 1, Tp...>(t);
  }
  
  /**
   * Sum up all layer activations by evaluating all layers.
   *
   * enable_if (SFINAE) is used to iterate through the network layers.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LayerForward(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LayerForward(std::tuple<Tp...>& t)
  {
    std::get<I>(t).FeedForward(
        std::get<I>(t).InputActivation(),
        std::get<I>(t).InputActivation());
    LayerForward<I + 1, Tp...>(t);
  }

  /*
   * Calculate the output error and update the overall error.
   */
  template<typename VecType, typename... Tp>
  double OutputError(std::tuple<Tp...>& t,
                   const VecType& target,
                   VecType& error)
  {
    // Calculate and store the output error.
    outputLayer.calculateError(std::get<0>(
        std::get<sizeof...(Tp) - 1>(t)).InputActivation(),
        target, error);

    // Masures the network's performance with the specified performance
    // function.
    return PerformanceFunction::error(std::get<0>(
        std::get<sizeof...(Tp) - 1>(t)).InputActivation(),
        target);
  }

  /**
   * Calculate and store the output activation.
   */
  template<typename VecType, typename... Tp>
  void OutputPrediction(std::tuple<Tp...>& t, VecType& output)
  {
    // Calculate and store the output prediction.
    outputLayer.outputClass(std::get<0>(
        std::get<sizeof...(Tp) - 1>(t)).InputActivation(),
        output);
  }

  /**
   * Run a single iteration of the feed backward algorithm, using the given
   * error of the output layer.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * connections, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 0, typename VecType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  FeedBackward(std::tuple<Tp...>& /* unused */, VecType& /* unused */) { }

  template<size_t I = 1, typename VecType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  FeedBackward(std::tuple<Tp...>& t, VecType& error)
  {
    // Pass initial error to the last layer.
    if (I == 1)
      std::get<0>(std::get<sizeof...(Tp) - I>(t)).Delta() = error;
    
    LayerBackward(std::get<sizeof...(Tp) - I>(t));
    ConnectionBackward(std::get<sizeof...(Tp) - I -1>(t));

    FeedBackward<I + 2, VecType, Tp...>(t, error);
  }

  /**
   * Back propagate the given error through layers.
   *
   * enable_if (SFINAE) is used to iterate through the network layers.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LayerBackward(std::tuple<Tp...>& /* unused */) { }
  
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LayerBackward(std::tuple<Tp...>& t)
  {
    std::get<I>(t).FeedBackward(std::get<I>(t).InputActivation(),
                                std::get<I>(t).Delta(),
                                std::get<I>(t).Delta());
    LayerBackward<I + 1, Tp...>(t);
  }
  
  /**
   * Back propagate the given error through connections.
   *
   * enable_if (SFINAE) is used to iterate through the network connections.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ConnectionBackward(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ConnectionBackward(std::tuple<Tp...>& t)
  {
    std::get<I>(t).FeedBackward(std::get<I>(t).OutputLayer().Delta());
    ConnectionBackward<I + 1, Tp...>(t);
  }

  /**
   * Helper function to iterate through all connection modules and to update
   * the gradient storage.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * connections, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I >= sizeof...(Tp), void>::type
  UpdateGradients(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  UpdateGradients(std::tuple<Tp...>& t)
  {
    Gradients(std::get<I>(t));
    UpdateGradients<I + 2, Tp...>(t);
  }

  /**
   * Sum up all gradients and store the results in the gradients storage.
   *
   * enable_if (SFINAE) is used to iterate through the network connections.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  Gradients(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  Gradients(std::tuple<Tp...>& t)
  {
    // A connection module must have WeightsDelta when applied in CNN.
    gradients[gradientNum++] += std::get<I>(t).WeightsDelta();
    Gradients<I + 1, Tp...>(t);
  }

  /**
   * Helper function to update the weights using the specified optimizer and
   * the given input.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * connections, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I >= sizeof...(Tp), void>::type
  ApplyGradients(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ApplyGradients(std::tuple<Tp...>& t)
  {
    Apply(std::get<I>(t));
    ApplyGradients<I + 2, Tp...>(t);
  }

  /**
   * Update the weights using the gradients from the gradient store.
   *
   * enable_if (SFINAE) is used to iterate through the network connections.
   * The general case peels off the first type and recurses, as usual with
   * variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  Apply(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  Apply(std::tuple<Tp...>& t)
  {
    if (seqNum > 1)
      gradients[gradientNum] /= seqNum;
    
    std::get<I>(t).Optimzer().UpdateWeights(std::get<I>(t).Weights(),
                                            gradients[gradientNum], trainError);
    
    // Reset the gradient storage.
    gradients[gradientNum++].zeros();
    Apply<I + 1, Tp...>(t);
  }

 /**
   * Helper function to iterate through all connection modules and to build
   * gradient storage.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * connections, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I >= sizeof...(Tp), void>::type
  InitLayer(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  InitLayer(std::tuple<Tp...>& t)
  {
    Layer(std::get<I>(t));
    InitLayer<I + 2, Tp...>(t);
  }

  /**
   * Iterate through all connections and build the the gradient storage.
   *
   * enable_if (SFINAE) is used to select between two template overloads of
   * the get function - one for when I is equal the size of the tuple of
   * connections, and one for the general case which peels off the first type
   * and recurses, as usual with variadic function templates.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  Layer(std::tuple<Tp...>& /* unused */) { }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  Layer(std::tuple<Tp...>& t)
  {
    gradients.push_back(
        new MatType(std::get<I>(t).Weights().n_rows,
        std::get<I>(t).Weights().n_cols, arma::fill::zeros));

    Layer<I + 1, Tp...>(t);
  }

  //! The connection modules used to build the network.
  ConnectionTypes network;

  //! The outputlayer used to evaluate the network
  OutputLayerType& outputLayer;

  //! The current training error of the network.
  double trainError;

  //! The gradient storage we are using to perform the feed backward pass.
  boost::ptr_vector<MatType> gradients;

  //! The index of the currently activate gradient.
  size_t gradientNum;

  //! The number of the current input sequence.
  size_t seqNum;
}; // class CNN


//! Network traits for the CNN network.
template <
    typename ConnectionTypes,
    typename OutputLayerType,
    class PerformanceFunction
>
class NetworkTraits<
  CNN<ConnectionTypes, OutputLayerType, PerformanceFunction> >
{
 public:
  static const bool IsFNN = false;
  static const bool IsRNN = false;
  static const bool IsCNN = true;
};

}; // namespace ann
}; // namespace mlpack

#endif
