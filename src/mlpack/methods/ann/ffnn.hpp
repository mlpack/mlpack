/**
 * @file ffnn.hpp
 * @author Marcus Edel
 *
 * Definition of the FFNN class, which implements feed forward neural networks.
 */
#ifndef __MLPACK_METHODS_ANN_FFNN_HPP
#define __MLPACK_METHODS_ANN_FFNN_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include <mlpack/methods/ann/network_traits.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard feed forward network.
 *
 * @tparam ConnectionTypes Tuple that contains all connection module which will
 * be used to construct the network.
 * @tparam OutputLayerType The outputlayer type used to evaluate the network.
 * @tparam PerformanceFunction Performance strategy used to claculate the error.
 * @tparam MaType of gradients. (arma::mat or arma::sp_mat).
 */
template <
  typename ConnectionTypes,
  typename OutputLayerType,
  class PerformanceFunction = CrossEntropyErrorFunction<>,
  typename MatType = arma::mat
>
class FFNN
{
  public:
    /**
     * Construct the FFNN object, which will construct a feed forward neural
     * network with the specified layers.
     *
     * @param network The network modules used to construct net network.
     * @param outputLayer The outputlayer used to evaluate the network.
     */
    FFNN(const ConnectionTypes& network, OutputLayerType& outputLayer)
        : network(network), outputLayer(outputLayer), err(0)
    {
      // Nothing to do here.
    }

    /**
     * Run a single iteration of the feed forward algorithm, using the given
     * input and target vector, updating the resulting error into the error
     * vector.
     *
     * @param input Input data used to evaluat the network.
     * @param target Target data used to calculate the network error.
     * @param error The calulated error of the output layer.
     * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
     */
    template <typename VecType>
    void FeedForward(const VecType& input,
                     const VecType& target,
                     VecType& error)
    {
      ResetActivations(network);
      std::get<0>(
            std::get<0>(network)).InputLayer().InputActivation() = input;

      FeedForward(network, target, error);
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
     * Updating the weights using the specified optimizer and the given input.
     *
     * @param input Input data used to evaluate the network.
     * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
     */
    template <typename VecType>
    void ApplyGradients(const VecType& /* unused */)
    {
      gradientNum = 0;
      ApplyGradients(network);

      // Reset the overall error.
      err = 0;
    }

    //! Get the error of the network.
    double Error() const { return trainError; }

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
      std::get<I>(t).OutputLayer().InputActivation().zeros();
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
    template<size_t I = 0,  typename VecType, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    FeedForward(std::tuple<Tp...>& t,
                const VecType& target,
                VecType& error)

    {
      // Calculate and store the output error.
      outputLayer.calculateError(std::get<0>(
          std::get<I - 1>(t)).OutputLayer().InputActivation(), target,
          error);

      // Masures the network's performance with the specified performance
      // function.
      err += PerformanceFunction::error(std::get<0>(
          std::get<I - 1>(t)).OutputLayer().InputActivation(), target);

      // Update the final training error.
      trainError = err;
    }

    template<size_t I = 0, typename VecType, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    FeedForward(std::tuple<Tp...>& t,
                const VecType& target,
                VecType& error)
    {
      Forward(std::get<I>(t));

      // Use the first connection to perform the feed forward algorithm.
      std::get<0>(std::get<I>(t)).OutputLayer().FeedForward(
          std::get<0>(std::get<I>(t)).OutputLayer().InputActivation(),
          std::get<0>(std::get<I>(t)).OutputLayer().InputActivation());

      FeedForward<I + 1, VecType, Tp...>(t, target, error);
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
      // the output layer use specified error vector to store the error and to
      // perform the feed backward pass.
      if (I == 1)
      {
        // Use the first connection from the last connection module to
        // calculate the error.
        std::get<0>(std::get<sizeof...(Tp) - I>(t)).OutputLayer().FeedBackward(
            std::get<0>(
            std::get<sizeof...(Tp) - I>(t)).OutputLayer().InputActivation(),
            error, std::get<0>(
            std::get<sizeof...(Tp) - I>(t)).OutputLayer().Delta());
      }

      Backward(std::get<sizeof...(Tp) - I>(t), std::get<0>(
          std::get<sizeof...(Tp) - I>(t)).OutputLayer().Delta());

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

      // We calculate the delta only for non bias layer.
      if (!LayerTraits<typename std::remove_reference<decltype(
          std::get<I>(t).InputLayer())>::type>::IsBiasLayer)
      {
        std::get<I>(t).InputLayer().FeedBackward(
            std::get<I>(t).InputLayer().InputActivation(),
            std::get<I>(t).Delta(), std::get<I>(t).InputLayer().Delta());
      }

      Backward<I + 1, VecType, Tp...>(t, error);
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
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    UpdateGradients(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    UpdateGradients(std::tuple<Tp...>& t)
    {
      Gradients(std::get<I>(t));
      UpdateGradients<I + 1, Tp...>(t);
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
      gradients[gradientNum++] += std::get<I>(t).OutputLayer().Delta() *
          std::get<I>(t).InputLayer().InputActivation().t();

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
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    ApplyGradients(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    ApplyGradients(std::tuple<Tp...>& t)
    {
      Apply(std::get<I>(t));
      ApplyGradients<I + 1, Tp...>(t);
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
      std::get<I>(t).Optimzer().UpdateWeights(std::get<I>(t).Weights(),
          gradients[gradientNum], err);

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
    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I == sizeof...(Tp), void>::type
    InitLayer(std::tuple<Tp...>& /* unused */) { }

    template<size_t I = 0, typename... Tp>
    typename std::enable_if<I < sizeof...(Tp), void>::type
    InitLayer(std::tuple<Tp...>& t)
    {
      Layer(std::get<I>(t));
      InitLayer<I + 1, Tp...>(t);
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
      gradients.push_back(new MatType(std::get<I>(t).Weights().n_rows,
          std::get<I>(t).Weights().n_cols, arma::fill::zeros));

      Layer<I + 1, Tp...>(t);
    }

    //! The connection modules used to build the network.
    ConnectionTypes network;

    //! The outputlayer used to evaluate the network
    OutputLayerType& outputLayer;

    //! The current error of the network.
    double err;

    //! The current training error of the network.
    double trainError;

    //! The gradient storage we are using to perform the feed backward pass.
    boost::ptr_vector<MatType> gradients;

    //! The index of the currently activate gradient.
    size_t gradientNum;
}; // class FFNN


//! Network traits for the FFNN network.
template <
  typename ConnectionTypes,
  typename OutputLayerType,
  class PerformanceFunction
>
class NetworkTraits<
    FFNN<ConnectionTypes, OutputLayerType, PerformanceFunction> >
{
 public:
  static const bool IsFNN = true;
  static const bool IsRNN = false;
};

}; // namespace ann
}; // namespace mlpack

#endif
