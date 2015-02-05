/**
 * @file neuron_layer_2d.hpp
 * @author Shangtong Zhang
 *
 * Definition of the NeuronLayer2D class, which implements a standard network
 * layer for 2-dimension data.
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_NEURON_LAYER_2D_HPP
#define __MLPACK_METHOS_ANN_LAYER_NEURON_LAYER_2D_HPP

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
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat
>
class NeuronLayer2D

{
 public:
  /**
   * Create the NeuronLayer2D object using the specified rows and columns.
   *
   * @param layerRows The number of rows of neurons.
   * @param layerCols The number of columns of neurons.
   */
  NeuronLayer2D(const size_t layerRows, const size_t layerCols) :
    layerRows(layerRows), layerCols(layerCols),
    localInputAcitvations(arma::ones<MatType>(layerRows, layerCols)),
    inputActivations(localInputAcitvations),
    localDelta(arma::zeros<MatType>(layerRows, layerCols)),
    delta(localDelta) {
    
  }
  
  /**
   * Create the NeuronLayer2D object using the specified inputActivations and delta.
   * This allow shared memory among layers, 
   * which make it easier to combine layers together in some special condition.
   *
   * @param inputActivations Outside storage for storing input activations.
   * @param delta Outside storage for storing delta, 
   *        the passed error in backward propagation.
   */
  NeuronLayer2D(MatType& inputActivations, MatType& delta) :
    layerRows(inputActivations.n_rows),
    layerCols(inputActivations.n_cols),
    inputActivations(inputActivations),
    delta(delta){
      
  }
  
  /**
   * Copy Constructor
   */
  NeuronLayer2D(const NeuronLayer2D& l) :
  layerRows(l.layerRows), layerCols(l.layerCols),
  localInputAcitvations(l.localInputAcitvations),
  inputActivations(l.localInputAcitvations.n_elem == 0 ?
                   l.inputActivations : localInputAcitvations),
  localDelta(l.localDelta),
  delta(l.localDelta.n_elem == 0 ? l.delta : localDelta){
    
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param inputActivation Input data used for evaluating the specified
   * activity function.
   * @param outputActivation Data to store the resulting output activation.
   */
  void FeedForward(const MatType& inputActivation, MatType& outputActivation)
  {
    Log::Debug << "NeuronLayer2D::FeedForward" << std::endl;
    Log::Debug << "Input:\n" << inputActivation << std::endl;
    ActivationFunction::fn(inputActivation, outputActivation);
    Log::Debug << "Output:\n" << outputActivation << std::endl;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param inputActivation Input data used for calculating the function f(x).
   * @param error The backpropagated error.
   * @param delta The passed error in backward propagation.
   */
  void FeedBackward(const MatType& inputActivation,
                    const MatType& error,
                    MatType& delta)
  {
    Log::Debug << "NeuronLayer2D::FeedBackward" << std::endl;
    Log::Debug << "Error:\n" << error << std::endl;
    MatType derivative;
    ActivationFunction::deriv(inputActivation, derivative);

    delta = error % derivative;
    Log::Debug << "Delta:\n" << delta << std::endl;
  }

  //! Get the input activations.
  MatType& InputActivation() const { return inputActivations; }
  //! Modify the input activations.
  MatType& InputActivation() { return inputActivations; }

  //! Get the error passed in backward propagation.
  MatType& Delta() const { return delta; }
  //! Modify the error passed in backward propagation.
  MatType& Delta() { return delta; }

  //! Get the number of layer rows.
  size_t LayerRows() const { return layerRows; }

  //! Get the number of layer colums.
  size_t LayerCols() const { return layerCols; }

 private:
  //! Locally-stored number of layer rows.
  size_t layerRows;
  
  //! Locally-stored number of layer cols.
  size_t layerCols;
  
  //! Locally-stored input activation object.
  MatType localInputAcitvations;
  
  //! Reference to locally-stored or outside input activation object.
  MatType& inputActivations;
  
  //! Locally-stored delta object.
  MatType localDelta;
  
  //! Reference to locally-stored or outside delta object.
  MatType& delta;
  

}; // class NeuronLayer2D

// Convenience typedefs.

/**
 * Standard Input-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat
>
using InputLayer2D = NeuronLayer2D<ActivationFunction, MatType>;

/**
 * Standard Hidden-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat
>
using HiddenLayer2D = NeuronLayer2D<ActivationFunction, MatType>;

/**
 * Layer of rectified linear units (relu) using the rectifier activation
 * function.
 */
template <
    class ActivationFunction = RectifierFunction,
    typename MatType = arma::mat
>
using ReluLayer2D = NeuronLayer2D<ActivationFunction, MatType>;


}; // namespace ann
}; // namespace mlpack

#endif
