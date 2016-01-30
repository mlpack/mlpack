/**
 * @file base_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BaseLayer class, which attaches various functions to the
 * embedding layer.
 */
#ifndef __MLPACK_METHODS_NN_LAYER_BASE_LAYER_HPP
#define __MLPACK_METHODS_NN_LAYER_BASE_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_autoencoder/activation_functions/logistic_function.hpp>
#include <mlpack/methods/sparse_autoencoder/activation_functions/rectifier_function.hpp>

namespace mlpack {
namespace nn /** Neural Network. */ {

/**
 * Implementation of the base layer. The base layer works as a metaclass which
 * attaches various functions to the embedding layer.
 *
 * A few convenience typedefs are given:
 *
 *  - SigmoidLayer
 *  - IdentityLayer
 *  - BaseLayer2D
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam InputDataType Type of the input data (arma::mat).
 * @tparam OutputDataType Type of the output data (arma::mat).
 */
template <
    class ActivationFunction = LogisticFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class BaseLayer
{
 public:
  using ActivateFunction = ActivationFunction;
  /**
   * Create the BaseLayer object.
   */
  BaseLayer()
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& input, OutputType& output)
  {
    ActivationFunction::fn(input, output);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& input,
                const DataType& gy,
                DataType& g)
  {
    DataType derivative;
    ActivationFunction::deriv(input, derivative);
    g = gy % derivative;
  }    
}; // class BaseLayer

// Convenience typedefs.

/**
 * Standard Sigmoid-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using SigmoidLayer = BaseLayer<
    ActivationFunction, InputDataType, OutputDataType>;

/**
 * RELU Layer using the rectifier activation function.
 */
template <
    class ActivationFunction = RectifierFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using RectifierLayer = BaseLayer<
    ActivationFunction, InputDataType, OutputDataType>;


}; // namespace nn
}; // namespace mlpack

#endif
