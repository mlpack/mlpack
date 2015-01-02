/**
 * @file neuron_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the NeuronLayer class, which implements a standard network
 * layer.
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_NEURON_LAYER_HPP
#define __MLPACK_METHOS_ANN_LAYER_NEURON_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard network layer.
 *
 * This class allows the specification of the type of the activation function.
 *
 * A few convenience typedefs are given:
 *
 *  - InputLayer
 *  - HiddenLayer
 *  - ReluLayer
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class NeuronLayer

{
 public:
  /**
   * Create the NeuronLayer object using the specified number of neurons.
   *
   * @param layerSize The number of neurons.
   */
  NeuronLayer(const size_t layerSize) :
      inputActivations(arma::zeros<VecType>(layerSize)),
      delta(arma::zeros<VecType>(layerSize)),
      layerSize(layerSize)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param inputActivation Input data used for evaluating the specified
   * activity function.
   * @param outputActivation Data to store the resulting output activation.
   */
  void FeedForward(const VecType& inputActivation, VecType& outputActivation)
  {
    ActivationFunction::fn(inputActivation, outputActivation);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param inputActivation Input data used for calculating the function f(x).
   * @param error The backpropagated error.
   * @param delta The calculating delta using the partial derivative of the
   * error with respect to a weight.
   */
  void FeedBackward(const VecType& inputActivation,
                    const VecType& error,
                    VecType& delta)
  {
    VecType derivative;
    ActivationFunction::deriv(inputActivation, derivative);

    delta = error % derivative;
  }

  //! Get the input activations.
  VecType& InputActivation() const { return inputActivations; }
  //  //! Modify the input activations.
  VecType& InputActivation() { return inputActivations; }

  //! Get the detla.
  VecType& Delta() const { return delta; }
 //  //! Modify the delta.
  VecType& Delta() { return delta; }

  //! Get input size.
  size_t InputSize() const { return layerSize; }
  //  //! Modify the delta.
  size_t& InputSize() { return layerSize; }

  //! Get output size.
  size_t OutputSize() const { return layerSize; }
  //! Modify the output size.
  size_t& OutputSize() { return layerSize; }

 private:
  //! Locally-stored input activation object.
  VecType inputActivations;

  //! Locally-stored delta object.
  VecType delta;

  //! Locally-stored number of neurons.
  size_t layerSize;
}; // class NeuronLayer

// Convenience typedefs.

/**
 * Standard Input-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
using InputLayer = NeuronLayer<ActivationFunction, MatType, VecType>;

/**
 * Standard Hidden-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
using HiddenLayer = NeuronLayer<ActivationFunction, MatType, VecType>;

/**
 * Layer of rectified linear units (relu) using the rectifier activation
 * function.
 */
template <
    class ActivationFunction = RectifierFunction,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
using ReluLayer = NeuronLayer<ActivationFunction, MatType, VecType>;


}; // namespace ann
}; // namespace mlpack

#endif
