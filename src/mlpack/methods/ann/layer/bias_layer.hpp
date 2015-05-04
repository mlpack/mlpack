/**
 * @file bias_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BiasLayer class, which implements a standard bias
 * layer.
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_BIAS_LAYER_HPP
#define __MLPACK_METHOS_ANN_LAYER_BIAS_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard bias layer with a default value of one.
 *
 * @tparam ActivationFunction Activation function used for the bias layer
 * (Default IdentityFunction).
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <
    class ActivationFunction = IdentityFunction,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class BiasLayer

{
 public:
  /**
   * Create the BiasLayer object using the specified number of bias units.
   *
   * @param layerSize The number of neurons.
   */
  BiasLayer(const size_t layerSize) :
      inputActivations(arma::ones<VecType>(layerSize)),
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
  const VecType& InputActivation() const { return inputActivations; }
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

  //! Get the number of layer slices.
  size_t LayerSlices() const { return 1; }

 private:
  //! Locally-stored input activation object.
  VecType inputActivations;

  //! Locally-stored delta object.
  VecType delta;

  //! Locally-stored number of neurons.
  size_t layerSize;
}; // class BiasLayer

//! Layer traits for the bias layer.
template<typename ActivationFunction, typename MatType, typename VecType>
class LayerTraits<BiasLayer<ActivationFunction, MatType, VecType> >
{
 public:
  /**
   * If true, then the layer is binary.
   */
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = true;
  static const bool IsLSTMLayer = false;
};

}; // namespace ann
}; // namespace mlpack

#endif
