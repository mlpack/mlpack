/**
 * @file rnn.hpp
 * @author Marcus Edel
 *
 * Definition of the RNN class, which implements recurrent neural networks.
 */
#ifndef __MLPACK_METHODS_ANN_RNN_HPP
#define __MLPACK_METHODS_ANN_RNN_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include <mlpack/methods/ann/network_traits.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a recurrent neural network.
 *
 * @tparam ConnectionTypes Tuple that contains all connection module which will
 * be used to construct the network.
 * @tparam OutputLayerType The outputlayer type used to evaluate the network.
 * @tparam PerformanceFunction Performance strategy used to claculate the error.
 * @tparam MaType of data (arma::mat or arma::sp_mat).
 * @tparam VecTypeDelta Type of the delta (arma::colvec, arma::mat or
 * arma::sp_mat).
 */
template <
  typename ConnectionTypes,
  typename OutputLayerType,
  class PerformanceFunction = CrossEntropyErrorFunction<>,
  typename MatType = arma::mat,
  typename VecTypeDelta = arma::colvec
>
class RNN
{
  public:
    /**
     * Construct the RNN object, which will construct a frecurrent neural
     * network with the specified layers.
     *
     * @param network The network modules used to construct net network.
     * @param outputLayer The outputlayer used to evaluate the network.
     */
    RNN(const ConnectionTypes& network, OutputLayerType& outputLayer) :
        network(network), outputLayer(outputLayer)
    {
      // Nothing to do here.
    }

    /**
     * Run a single iteration of the feed forward algorithm, using the given
     * input and target vector, updating the resulting error into the error
     * vector.
     *
     * @param input Input data used for evaluating the network.
     * @param target Target data used to calculate the network error.
     * @param error The calulated error of the output layer.
     * @tparam VecType Type of target data (arma::colvec, arma::mat or
     * arma::sp_mat).
     */
    template <typename VecType>
    void FeedForward(const MatType& input,
                     const VecType& target,
                     MatType& error)
    {
      // Initialize the activation storage only once.
      if (!activations.size())
        InitLayer(network, input, target);

      // Reset the overall error.
      err = 0;
      error = MatType(target.n_elem, input.n_rows);

      // Iterate through the input sequence and perform the feed forward pass.
      for (seqNum = 0; seqNum < input.n_rows; seqNum++)
      {
        // Reset the network by zeroing the layer activations and set the input
        // activation.
        ResetActivations(network);
        std::get<0>(std::get<0>(
            network)).InputLayer().InputActivation() = input(seqNum);

        arma::colvec seqError = error.unsafe_col(seqNum);
        FeedForward(network, target, seqError);

        // Save the network activation for the backward pass.
        if (seqNum < (input.n_rows - 1))
        {
          layerNum = 0;
          SaveActivations(network);
        }
      }
    }

    /**
     * Run a single iteration of the feed backward algorithm, using the given
     * error of the output layer.
     *
     * @param error The calulated error of the output layer.
     */
    void FeedBackward(const MatType& error)
    {
      // Reset the network gradients by zeroing the storage.
      for (size_t i = 0; i < gradients.size(); ++i)
        gradients[i].zeros();

      // Reset the network deltas by zeroing the storage.
      for (size_t i = 0; i < delta.size(); ++i)
        delta[i].zeros();

      // Iterate through the input sequence and perform the feed backward pass.
      for (seqNum = error.n_cols - 1; seqNum >= 0; seqNum--)
      {
        gradientNum = 0;
        FeedBackward(network, error.unsafe_col(seqNum));

        // Load the network activation for the upcoming backward pass.
        if (seqNum > 0)
        {
          layerNum = 0;
          LoadActivations(network);
        }
      }
    }

    /**
     * Updating the weights using the specified optimizer and the given input.
     *
     * @param input Input data used for evaluating the network.
     * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
     */
    template <typename VecType>
    void ApplyGradients(const VecType& input)
    {
      gradientNum = 0;
      ApplyGradients(network, input);
    }

  private:
    /**
     * Helper function to reset the network by zeroing the layer activations.
     *
     * enable_if (SFINAE) is used to iterate through the network connection
     * modules. The general case peels off the first type and recurses, as usual
     * with variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    ResetActivations(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    ResetActivations(std::tuple<Tp...>& t)
    {
      Reset(std::get<I>(t));
      ResetActivations<I + 1, Tp...>(t);
    }

    /**
     * Reset the network by zeroing the layer activations.
     *
     * enable_if (SFINAE) is used to iterate through the network connections.
     * The general case peels off the first type and recurses, as usual with
     * variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    Reset(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    Reset(std::tuple<Tp...>& t)
    {
      std::get<I>(t).OutputLayer().InputActivation().zeros(
          std::get<I>(t).OutputLayer().InputSize());
      Reset<I + 1, Tp...>(t);
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
    template<size_t I = 0,
             typename TargetVecType,
             typename ErrorVecType,
             typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    FeedForward(std::tuple<Tp...>& t,
                TargetVecType& target,
                ErrorVecType& error)
    {
      // Calculate and store the output error.
      outputLayer.calculateError(std::get<0>(
          std::get<I - 1>(t)).OutputLayer().InputActivation(), target,
          error);

      // Save the output activation for the upcoming feed backward pass.
      activations.back().unsafe_col(seqNum) = std::get<0>(
          std::get<I - 1>(t)).OutputLayer().InputActivation();

      // Masures the network's performance with the specified performance
      // function.
      err = PerformanceFunction::error(std::get<0>(
          std::get<I - 1>(t)).OutputLayer().InputActivation(), target);
    }

    template<size_t I = 0,
            typename TargetVecType,
            typename ErrorVecType,
            typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    FeedForward(std::tuple<Tp...>& t,
                TargetVecType& target,
                ErrorVecType& error)
    {
      Forward(std::get<I>(t));

      // Use the first connection to perform the feed forward algorithm.
      std::get<0>(std::get<I>(t)).OutputLayer().FeedForward(
          std::get<0>(std::get<I>(t)).OutputLayer().InputActivation(),
          std::get<0>(std::get<I>(t)).OutputLayer().InputActivation());

      FeedForward<I + 1, TargetVecType, ErrorVecType, Tp...>(t, target, error);
    }

    /**
     * Sum up all layer activations by evaluating all connections.
     *
     * enable_if (SFINAE) is used to iterate through the network connections.
     * The general case peels off the first type and recurses, as usual with
     * variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    Forward(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    Forward(std::tuple<Tp...>& t)
    {
      std::get<I>(t).FeedForward(std::get<I>(t).InputLayer().InputActivation());
      Forward<I + 1, Tp...>(t);
    }

    /**
     * Run a single iteration of the feed backward algorithm, using the given
     * error of the output layer. Note that we iterate backward through the
     * connection modules.
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
      // Distinguish between the output layer and the other layer. In case of
      // the output layer use the specified error vector to store the error and
      // to perform the feed backward pass.
      if (I == 1)
      {
        // Use the first connection from the last connection module to
        // calculate the error.
        std::get<0>(std::get<sizeof...(Tp) - I>(t)).OutputLayer().FeedBackward(
            activations.back().unsafe_col(seqNum), error,
            std::get<0>(std::get<sizeof...(Tp) - I>(t)).OutputLayer().Delta());

        // Save the delta for the upcoming feed backward pass.
        delta.back() += std::get<0>(
            std::get<sizeof...(Tp) - I>(t)).OutputLayer().Delta();

        // Save the gradient to update the weights at the end.
        gradients.back() += std::get<0>(
            std::get<sizeof...(Tp) - I>(t)).OutputLayer().Delta() *
            std::get<0>(
            std::get<sizeof...(Tp) - I>(t)).InputLayer().InputActivation().t();
      }

      Backward(std::get<sizeof...(Tp) - I>(t), delta[delta.size() - I]);
      UpdateGradients(std::get<sizeof...(Tp) - I - 1>(t));

      FeedBackward<I + 1, VecType, Tp...>(t, error);
    }

    /**
     * Back propagate the given error and store the delta in the connection
     * between the corresponding layer.
     *
     * enable_if (SFINAE) is used to iterate through the network connections.
     * The general case peels off the first type and recurses, as usual with
     * variadic function templates.
     */
    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    Backward(std::tuple<Tp...>& /* unused */, VecType& /* unused */) { }

    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    Backward(std::tuple<Tp...>& t, VecType& error)
    {
      std::get<I>(t).FeedBackward(error);

      // We calculate the delta only for non bias layer and self connections.
      if (!(ConnectionTraits<typename std::remove_reference<decltype(
            std::get<I>(t))>::type>::IsSelfConnection ||
        LayerTraits<typename std::remove_reference<decltype(
            std::get<I>(t).InputLayer())>::type>::IsBiasLayer ||
        ConnectionTraits<typename std::remove_reference<decltype(
            std::get<I>(t))>::type>::IsFullselfConnection))
      {
        std::get<I>(t).InputLayer().FeedBackward(
            std::get<I>(t).InputLayer().InputActivation(),
            std::get<I>(t).Delta(), std::get<I>(t).InputLayer().Delta());
      }

      Backward<I + 1, VecType, Tp...>(t, error);
    }

    /**
     * Sum up all gradients and store the  results in the gradients storage.
     *
     * enable_if (SFINAE) is used to iterate through the network connections.
     * The general case peels off the first type and recurses, as usual with
     * variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    UpdateGradients(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    UpdateGradients(std::tuple<Tp...>& t)
    {
      gradients[gradientNum++] += std::get<I>(t).OutputLayer().Delta() *
          std::get<I>(t).InputLayer().InputActivation().t();

      UpdateGradients<I + 1, Tp...>(t);
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
    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp) - 1, void>::type
    ApplyGradients(std::tuple<Tp...>& /* unused */,
                   const VecType& /* unused */) { }

    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp) - 1, void>::type
    ApplyGradients(std::tuple<Tp...>& t, const VecType& input)
    {
      Gradients(std::get<I>(t));
      ApplyGradients<I + 1, VecType, Tp...>(t, input);
    }

    /**
     * Update the weights using the specified optimizer,the given input and the
     * calculated delta.
     *
     * enable_if (SFINAE) is used to select between two template overloads of
     * the get function - one for when I is equal the size of the tuple of
     * connections, and one for the general case which peels off the first type
     * and recurses, as usual with variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    Gradients(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    Gradients(std::tuple<Tp...>& t)
    {
      std::get<I>(t).Optimzer().UpdateWeights(std::get<I>(t).Weights(),
          gradients[gradientNum++], err);

      Gradients<I + 1, Tp...>(t);
    }

    /**
     * Helper function to iterate through all connection modules and to build
     * the activation, gradients and delta storage.
     *
     * enable_if (SFINAE) is used to select between two template overloads of
     * the get function - one for when I is equal the size of the tuple of
     * connections, and one for the general case which peels off the first type
     * and recurses, as usual with variadic function templates.
     */
    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    InitLayer(std::tuple<Tp...>& /* unused */,
              const MatType& input,
              const VecType& target)
    {
      activations.push_back(new MatType(target.n_elem, input.n_elem));
    }

    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    InitLayer(std::tuple<Tp...>& t, const MatType& input, const VecType& target)
    {
      Layer(std::get<I>(t), input);
      InitLayer<I + 1, VecType, Tp...>(t, input, target);
    }

    /**
     * Iterate through all connections and build the the activation, gradients
     * and delta storage.
     *
     * enable_if (SFINAE) is used to select between two template overloads of
     * the get function - one for when I is equal the size of the tuple of
     * connections, and one for the general case which peels off the first type
     * and recurses, as usual with variadic function templates.
     */
    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    Layer(std::tuple<Tp...>& /* unused */, const VecType& /* unused */) { }

    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    Layer(std::tuple<Tp...>& t, const VecType& input)
    {
      activations.push_back(new MatType(
        std::get<I>(t).InputLayer().OutputSize(), input.n_elem));

      gradients.push_back(new MatType(std::get<I>(t).Weights().n_rows,
          std::get<I>(t).Weights().n_cols));

      // We calculate the delta only for non bias layer and self connections.
      if (!(ConnectionTraits<typename std::remove_reference<decltype(
              std::get<I>(t))>::type>::IsSelfConnection ||
          LayerTraits<typename std::remove_reference<decltype(
              std::get<I>(t).InputLayer())>::type>::IsBiasLayer ||
          ConnectionTraits<typename std::remove_reference<decltype(
            std::get<I>(t))>::type>::IsFullselfConnection))
      {
        delta.push_back(new VecTypeDelta(std::get<I>(t).Weights().n_rows));
      }

      Layer<I + 1, VecType, Tp...>(t, input);
    }

    /**
     * Helper function to iterate through all connection modules and to load
     * the layer activations.
     *
     * enable_if (SFINAE) is used to iterate through the network connection
     * modules. The general case peels off the first type and recurses, as usual
     * with variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    LoadActivations(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    LoadActivations(std::tuple<Tp...>& t)
    {
      Load(std::get<I>(t));
      LoadActivations<I + 1, Tp...>(t);
    }

    /**
     * Load and set the network layer activations.
     *
     * enable_if (SFINAE) is used to iterate through the network connections.
     * The general case peels off the first type and recurses, as usual with
     * variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    Load(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    Load(std::tuple<Tp...>& t)
    {
      std::get<I>(t).InputLayer().InputActivation() =
          activations[layerNum++].unsafe_col(seqNum - 1);
      Load<I + 1, Tp...>(t);
    }

    /**
     * Helper function to iterate through all connection modules and to save
     * the layer activations.
     *
     * enable_if (SFINAE) is used to iterate through the network connection
     * modules. The general case peels off the first type and recurses, as usual
     * with variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    SaveActivations(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    SaveActivations(std::tuple<Tp...>& t)
    {
      Save(std::get<I>(t));
      SaveActivations<I + 1, Tp...>(t);
    }

    /**
     * Save the network layer activations.
     *
     * enable_if (SFINAE) is used to iterate through the network connections.
     * The general case peels off the first type and recurses, as usual with
     * variadic function templates.
     */
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    Save(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    Save(std::tuple<Tp...>& t)
    {
      activations[layerNum++].unsafe_col(seqNum) =
          std::get<I>(t).InputLayer().InputActivation();

      // Use the activation from the corresponding outputlayer for
      // self connections.
      if (ConnectionTraits<typename std::remove_reference<decltype(
              std::get<I>(t))>::type>::IsSelfConnection ||
          ConnectionTraits<typename std::remove_reference<decltype(
              std::get<I>(t))>::type>::IsFullselfConnection)
      {
        std::get<I>(t).InputLayer().InputActivation() =
            std::get<I>(t).OutputLayer().InputActivation();
      }

      Save<I + 1, Tp...>(t);
    }

    //! The current error of the network.
    double err;

    //! The activation storage we are using to perform the feed backward pass.
    boost::ptr_vector<MatType> activations;

    //! The gradient storage we are using to perform the feed backward pass.
    boost::ptr_vector<MatType> gradients;

    //! The detla storage we are using to perform the feed backward pass.
    boost::ptr_vector<VecTypeDelta> delta;

    //! The index of the current sequence number.
    long int seqNum;

    //! The index of the currently activate layer.
    size_t layerNum;

    //! The index of the currently activate gradient.
    size_t gradientNum;

    //! The layer we are using to build the network.
    ConnectionTypes network;

    //! The outputlayer used to evaluate the network
    OutputLayerType& outputLayer;
}; // class RNN

//! Network traits for the FFNN network.
template <
  typename ConnectionTypes,
  typename OutputLayerType,
  class PerformanceFunction
>
class NetworkTraits<RNN<ConnectionTypes, OutputLayerType, PerformanceFunction> >
{
 public:
  static const bool IsFNN = false;
  static const bool IsRNN = true;
};

}; // namespace ann
}; // namespace mlpack

#endif

