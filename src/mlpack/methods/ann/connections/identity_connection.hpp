/**
 * @file identity_connection.hpp
 * @author Marcus Edel
 *
 * Implementation of the identity connection between the input- and output
 * layer.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_IDENTITY_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_IDENTITY_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

namespace mlpack{
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the indentity connection class. The identity connection
 * connects the i'th element from the input layer with i'th element of the
 * output layer.
 *
 * @tparam InputLayerType Type of the connected input layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam DataType Type of data (arma::mat, arma::sp_mat or arma::cube).
 */
template<
  typename InputLayerType,
  typename OutputLayerType,
  template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
  typename DataType = arma::colvec
>
class IdentityConnection
{
 public:
  /**
   * Create the IdentityConnection object using the specified input layer and
   * output layer.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   */
  IdentityConnection(InputLayerType& inputLayer,
                    OutputLayerType& outputLayer) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(0),
      weights(0)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network.
   *
   * @param input Input data used for the forward pass.
   */
  template<typename InputType>
  void FeedForward(const InputType& input)
  {
    outputLayer.InputActivation() += input;
  }

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param error The backpropagated error.
   */
  template<typename ErrorType>
  void FeedBackward(const ErrorType& error)
  {
    delta = error;
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param gradient The calculated gradient.
   */
  template<typename GradientType>
  void Gradient(GradientType& /* unused */)
  {
    // Nothing to do here.
  }

  //! Get the weights.
  DataType& Weights() const { return *weights; }
  //! Modify the weights.
  DataType& Weights() { return *weights; }

  //! Get the input layer.
  InputLayerType& InputLayer() const { return inputLayer; }
  //! Modify the input layer.
  InputLayerType& InputLayer() { return inputLayer; }

  //! Get the output layer.
  OutputLayerType& OutputLayer() const { return outputLayer; }
  //! Modify the output layer.
  OutputLayerType& OutputLayer() { return outputLayer; }

  //! Get the optimizer.
  OptimizerType<IdentityConnection<InputLayerType,
                                  OutputLayerType,
                                  OptimizerType,
                                  DataType>, DataType>& Optimzer() const
  {
    return *optimizer;
  }
  //! Modify the optimzer.
  OptimizerType<IdentityConnection<InputLayerType,
                                  OutputLayerType,
                                  OptimizerType,
                                  DataType>, DataType>& Optimzer()
  {
    return *optimizer;
  }

  //! Get the passed error in backward propagation.
  DataType& Delta() const { return delta; }
  //! Modify the passed error in backward propagation.
  DataType& Delta() { return delta; }

 private:
  //! Locally-stored input layer.
  InputLayerType& inputLayer;

  //! Locally-stored output layer.
  OutputLayerType& outputLayer;

  //! Locally-stored optimizer.
  OptimizerType<IdentityConnection<InputLayerType,
                                  OutputLayerType,
                                  OptimizerType,
                                  DataType>, DataType>* optimizer;

  //! Locally-stored weight object.
  DataType* weights;

  //! Locally-stored passed error in backward propagation.
  DataType delta;
}; // IdentityConnection class.

//! Connection traits for the identity connection.
template<
    typename InputLayerType,
    typename OutputLayerType,
    template<typename, typename> class OptimizerType,
    typename DataType
>
class ConnectionTraits<
    IdentityConnection<InputLayerType, OutputLayerType,OptimizerType,
    DataType> >
{
 public:
  static const bool IsSelfConnection = false;
  static const bool IsFullselfConnection = false;
  static const bool IsPoolingConnection = false;
  static const bool IsIdentityConnection = true;
};

}; // namespace ann
}; // namespace mlpack

#endif