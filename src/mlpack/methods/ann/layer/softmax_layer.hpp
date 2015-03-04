/**
 * @file softmax_layer.hpp
 * @author Marcus Edel
 * @author Shangtong Zhang
 *
 * Definition of the SoftmaxLayer class, which implements a standard network
 * layer.
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_SOFTMAX_LAYER_HPP
#define __MLPACK_METHOS_ANN_LAYER_SOFTMAX_LAYER_HPP

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
      layerSize(layerSize),
      localInputActivations(arma::zeros<VecType>(layerSize)),
      inputActivations(localInputActivations),
      localDelta(arma::zeros<VecType>(layerSize)),
      delta(localDelta)
  {
    // Nothing to do here.
  }
  
  /**
   * Create the SoftmaxLayer object using the specified inputActivations and delta.
   * This allow shared memory among layers,
   * which make it easier to combine layers together in some special condition.
   *
   * @param inputActivations Outside storage for storing input activations.
   * @param delta Outside storage for storing delta,
   *        the passed error in backward propagation.
   */
  SoftmaxLayer(VecType& inputActivations, VecType& delta) :
      layerSize(inputActivations.n_elem),
      inputActivations(inputActivations),
      delta(delta)
  {
    
  }
  
  /**
   * Copy Constructor
   */
  SoftmaxLayer(const SoftmaxLayer& l) :
  layerSize(l.layerSize),
  localInputActivations(l.localInputActivations),
  inputActivations(l.localInputActivations.elem == 0 ?
                   l.inputActivations : localInputActivations),
  localDelta(l.localDelta),
  delta(l.localDelta.elem == 0 ? l.delta : localDelta) {
    
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
    Log::Debug << "SoftmaxLayer::FeedForward" << std::endl;
    Log::Debug << "Input:\n" << inputActivation << std::endl;
    outputActivation = arma::trunc_exp(inputActivation);
    outputActivation /= arma::accu(outputActivation);
    Log::Debug << "Output:\n" << outputActivation << std::endl;
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
    Log::Debug << "SoftmaxLayer::FeedBackward" << std::endl;
    Log::Debug << "Input:\n" << error << std::endl;
    delta = error;
    Log::Debug << "Output:\n" << delta << std::endl;
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

 private:
  //! Locally-stored number of neurons.
  size_t layerSize;
  
  //! Locally-stored input activation object.
  VecType localInputActivations;
  
  //! Reference to locall-stored or outside input activation object.
  VecType& inputActivations;

  //! Locally-stored delta object.
  VecType localDelta;
  
  //! Reference to locally-stored or outside delta object.
  VecType& delta;
  
}; // class SoftmaxLayer

}; // namespace ann
}; // namespace mlpack

#endif
