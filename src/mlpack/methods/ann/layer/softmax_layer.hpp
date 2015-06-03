/**
 * @file softmax_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the SoftmaxLayer class, which implements a standard softmax
 * network layer.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_SOFTMAX_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_SOFTMAX_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard softmax layer.
 *
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <typename MatType = arma::mat, typename VecType = arma::colvec>
class SoftmaxLayer

{
 public:
  /**
   * Create the SoftmaxLayer object using the specified number of neurons.
   *
   * @param layerSize The number of neurons.
   */
  SoftmaxLayer(const size_t layerSize) :
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
    outputActivation = arma::trunc_exp(inputActivation);
    outputActivation /= arma::accu(outputActivation);
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
  void FeedBackward(const VecType& /* unused */,
                    const VecType& error,
                    VecType& delta)
  {
    delta = error;
  }

  //! Get the input activations.
  VecType& InputActivation() const { return inputActivations; }
  //! Modify the input activations.
  VecType& InputActivation() { return inputActivations; }

  //! Get the detla.
  VecType& Delta() const { return delta; }
  //! Modify the delta.
  VecType& Delta() { return delta; }

  //! Get input size.
  size_t InputSize() const { return layerSize; }
  //! Modify the delta.
  size_t& InputSize() { return layerSize; }

  //! Get output size.
  size_t OutputSize() const { return layerSize; }
  //! Modify the output size.
  size_t& OutputSize() { return layerSize; }

  //! Get the number of layer slices.
  size_t LayerSlices() const { return 1; }

  //! Get the number of output maps.
  size_t OutputMaps() const { return 1; }

 private:
  //! Locally-stored input activation object.
  VecType inputActivations;

  //! Locally-stored delta object.
  VecType delta;

  //! Locally-stored number of neurons.
  size_t layerSize;
}; // class SoftmaxLayer

}; // namespace ann
}; // namespace mlpack

#endif
