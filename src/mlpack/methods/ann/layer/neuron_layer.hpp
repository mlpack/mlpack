/**
 * @file neuron_layer.hpp
 * @author Marcus Edel
 * @author Shangtong Zhang
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
 * @tparam DataType Type of data (arma::colvec, arma::mat or arma::sp_mat,
 * arma::cube).
 */
template <
    class ActivationFunction = LogisticFunction,
    typename DataType = arma::colvec
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
      inputActivations(arma::zeros<DataType>(layerSize)),
      delta(arma::zeros<DataType>(layerSize)),
      layerRows(layerSize),
      layerSlices(1)
  {
    // Nothing to do here.
  }

  NeuronLayer(const size_t layerRows, const size_t layerCols) :
      inputActivations(arma::zeros<DataType>(layerRows, layerCols)),
      delta(arma::zeros<DataType>(layerRows, layerCols)),
      layerRows(layerRows),
      layerCols(layerCols),
      layerSlices(1)
  {
    // Nothing to do here.
  }

  NeuronLayer(const size_t layerRows,
              const size_t layerCols,
              const size_t layerSlices) :
      inputActivations(arma::zeros<DataType>(layerRows, layerCols, layerSlices)),
      delta(arma::zeros<DataType>(layerRows, layerCols, layerSlices)),
      layerRows(layerRows),
      layerCols(layerCols),
      layerSlices(layerSlices)
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
  void FeedForward(const DataType& inputActivation,
                   DataType& outputActivation)
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
  void FeedBackward(const DataType& inputActivation,
                    const DataType& error,
                    DataType& delta)
  {
    DataType derivative;
    ActivationFunction::deriv(inputActivation, derivative);
    delta = error % derivative;
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
  template<typename eT>
  void FeedBackward(const arma::Cube<eT>& inputActivation,
                    const arma::Mat<eT>& error,
                    arma::Cube<eT>& delta)
  {
    DataType derivative;
    ActivationFunction::deriv(inputActivation, derivative);
    delta = arma::cube(error.memptr(), inputActivation.n_rows,
        inputActivation.n_cols, inputActivation.n_slices) % derivative;
  }


  //! Get the input activations.
  DataType& InputActivation() const { return inputActivations; }
  //  //! Modify the input activations.
  DataType& InputActivation() { return inputActivations; }

  //! Get the detla.
  DataType& Delta() const { return delta; }
  //! Modify the delta.
  DataType& Delta() { return delta; }

  //! Get input size.
  size_t InputSize() const { return layerRows; }
  //! Modify the delta.
  size_t& InputSize() { return layerRows; }

  //! Get output size.
  size_t OutputSize() const { return layerRows; }
  //! Modify the output size.
  size_t& OutputSize() { return layerRows; }

  //! Get the number of layer rows.
  size_t LayerRows() const { return layerRows; }
  //! Modify the number of layer rows.
  size_t& LayerRows() { return layerRows; }

  //! Get the number of layer columns.
  size_t LayerCols() const { return layerCols; }
  //! Modify the number of layer columns.
  size_t& LayerCols() { return layerCols; }

  //! Get the number of layer slices.
  size_t LayerSlices() const { return layerSlices; }

 private:
  //! Locally-stored input activation object.
  DataType inputActivations;

  //! Locally-stored delta object.
  DataType delta;

  //! Locally-stored number of layer rows.
  size_t layerRows;

  //! Locally-stored number of layer cols.
  size_t layerCols;

  //! Locally-stored number of layer slices.
  size_t layerSlices;
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
