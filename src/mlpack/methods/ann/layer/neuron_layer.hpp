/**
 * @file neuron_layer.hpp
 * @author Marcus Edel
 * @author Shangtong Zhang
 *
 * Definition of the NeuronLayer class, which implements a standard network
 * layer for 1-dimensional or 2-dimensional data.
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
 * @tparam DataType Type of data (arma::mat or arma::colvec).
 */
template <
    class ActivationFunction = LogisticFunction,
    typename DataType = arma::colvec
>
class NeuronLayer
{
 public:
  /**
   * Create 2-dimensional NeuronLayer object using the specified rows and columns.
   * In this case, DataType must be aram::mat or other matrix type.
   *
   * @param layerRows The number of rows of neurons.
   * @param layerCols The number of columns of neurons.
   */
  NeuronLayer(const size_t layerRows, const size_t layerCols) :
      layerRows(layerRows), layerCols(layerCols),
      localInputAcitvations(arma::ones<DataType>(layerRows, layerCols)),
      inputActivations(localInputAcitvations),
      localDelta(arma::zeros<DataType>(layerRows, layerCols)),
      delta(localDelta)
  {
    // Nothing to do.
  }
  
  /**
   * Create 2-dimensional NeuronLayer object using the specified inputActivations and delta.
   * This allow shared memory among layers, 
   * which make it easier to combine layers together in some special condition.
   *
   * @param inputActivations Outside storage for storing input activations.
   * @param delta Outside storage for storing delta, 
   *        the passed error in backward propagation.
   */
  NeuronLayer(DataType& inputActivations, DataType& delta) :
      layerRows(inputActivations.n_rows),
      layerCols(inputActivations.n_cols),
      inputActivations(inputActivations),
      delta(delta)
  {
    // Nothing to do.
  }
  
  /**
   * Create 1-dimensional NeuronLayer object using the specified layer size.
   * In this case, DataType must be aram::colvec or other vector type.
   *
   * @param layerSize The number of neurons.
   */
  NeuronLayer(const size_t layerSize) :
      layerRows(layerSize), layerCols(1),
      localInputAcitvations(arma::ones<DataType>(layerRows)),
      inputActivations(localInputAcitvations),
      localDelta(arma::zeros<DataType>(layerRows)),
      delta(localDelta)
  {
    // Nothing to do.
  }
  
  /**
   * Copy Constructor
   */
  NeuronLayer(const NeuronLayer& l) :
      layerRows(l.layerRows), layerCols(l.layerCols),
      localInputAcitvations(l.localInputAcitvations),
      inputActivations(l.localInputAcitvations.n_elem == 0 ?
                       l.inputActivations : localInputAcitvations),
      localDelta(l.localDelta),
      delta(l.localDelta.n_elem == 0 ? l.delta : localDelta)
  {
    // Nothing to do.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param inputActivation Input data used for evaluating the specified
   * activity function.
   * @param outputActivation Data to store the resulting output activation.
   */
  void FeedForward(const DataType& inputActivation, DataType& outputActivation)
  {
    Log::Debug << "NeuronLayer::FeedForward" << std::endl;
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
  void FeedBackward(const DataType& inputActivation,
                    const DataType& error,
                    DataType& delta)
  {
    Log::Debug << "NeuronLayer::FeedBackward" << std::endl;
    Log::Debug << "Error:\n" << error << std::endl;
    DataType derivative;
    ActivationFunction::deriv(inputActivation, derivative);

    delta = error % derivative;
    Log::Debug << "Delta:\n" << delta << std::endl;
  }

  //! Get the input activations.
  DataType& InputActivation() const { return inputActivations; }
  //! Modify the input activations.
  DataType& InputActivation() { return inputActivations; }

  //! Get the error passed in backward propagation.
  DataType& Delta() const { return delta; }
  //! Modify the error passed in backward propagation.
  DataType& Delta() { return delta; }

  //! Get the number of layer rows.
  size_t LayerRows() const { return layerRows; }

  //! Get the number of layer colums.
  size_t LayerCols() const { return layerCols; }
  
  /**
   * Get the number of layer size.
   * Only for 1-dimsenional type.
   */
  size_t InputSize() const { return layerRows; }
  
  /**
   * Get the number of lyaer size.
   * Only for 1-dimsenional type.
   */
  size_t OutputSize() const { return layerRows; }

 private:
  //! Locally-stored number of layer rows.
  size_t layerRows;
  
  //! Locally-stored number of layer cols.
  size_t layerCols;
  
  //! Locally-stored input activation object.
  DataType localInputAcitvations;
  
  //! Reference to locally-stored or outside input activation object.
  DataType& inputActivations;
  
  //! Locally-stored delta object.
  DataType localDelta;
  
  //! Reference to locally-stored or outside delta object.
  DataType& delta;
  

}; // class NeuronLayer

// Convenience typedefs.

/**
 * Standard Input-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename DataType = arma::colvec
>
using InputLayer = NeuronLayer<ActivationFunction, DataType>;

/**
 * Standard Hidden-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename DataType = arma::colvec
>
using HiddenLayer = NeuronLayer<ActivationFunction, DataType>;

/**
 * Layer of rectified linear units (relu) using the rectifier activation
 * function.
 */
template <
    class ActivationFunction = RectifierFunction,
    typename DataType = arma::colvec
>
using ReluLayer = NeuronLayer<ActivationFunction, DataType>;


}; // namespace ann
}; // namespace mlpack

#endif
