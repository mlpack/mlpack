/**
 * @file base_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BaseLayer class, which attaches various functions to the
 * embedding layer.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the base layer. The base layer works as a metaclass which
 * attaches various functions to the embedding layer.
 *
 * A few convenience typedefs are given:
 *
 *  - SigmoidLayer
 *  - IdentityLayer
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam DataType Type of data (arma::colvec, arma::mat arma::sp_mat or
 * arma::cube).
 */
template <
    class ActivationFunction = LogisticFunction,
    typename DataType = arma::colvec
>
class BaseLayer
{
 public:
  /**
   * Create the BaseLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  BaseLayer(const size_t inSize, const size_t outSize) :
      inSize(inSize),
      outSize(outSize)
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
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    ActivationFunction::fn(input, output);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    arma::Mat<eT> derivative;
    ActivationFunction::deriv(input, derivative);
    g = gy % derivative;
  }

  //! Get the parameter.
  DataType& Parameter() const {return parameter; }
  //! Modify the parameter.
  DataType& Parameter() { return parameter; }

  //! Get the delta.
  DataType& Delta() const {return delta; }
  //! Modify the delta.
  DataType& Delta() { return delta; }

 private:
  //! Locally-stored number of input units.
  const size_t inSize;

  //! Locally-stored number of output units.
  const size_t outSize;

  //! Locally-stored delta object.
  DataType delta;

  //! Locally-stored parameter object.
  DataType parameter;
}; // class BaseLayer

// Convenience typedefs.

/**
 * Standard Sigmoid-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename DataType = arma::colvec
>
using SigmoidLayer = BaseLayer<ActivationFunction, DataType>;

/**
 * Standard Identity-Layer using the identity activation function.
 */
template <
    class ActivationFunction = IdentityFunction,
    typename DataType = arma::colvec
>
using IdentityLayer = BaseLayer<ActivationFunction, DataType>;


}; // namespace ann
}; // namespace mlpack

#endif
