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
 * @tparam DataType Type of data (arma::colvec, arma::mat arma::sp_mat or
 * arma::cube).
 */
template <typename DataType = arma::colvec>
class SoftmaxLayer
{
 public:
  /**
   * Create the SoftmaxLayer object using the specified number of neurons.
   *
   * @param layerSize The number of neurons.
   */
  SoftmaxLayer(const size_t layerSize) :
      inputActivations(arma::zeros<DataType>(layerSize)),
      delta(arma::zeros<DataType>(layerSize)),
      layerRows(layerSize),
      layerCols(1),
      layerSlices(1),
      outputMaps(1)
  {
    // Nothing to do here.
  }

  /**
   * Create 2-dimensional SoftmaxLayer object using the specified rows and
   * columns. In this case, DataType must be arma::mat or arma::sp_mat.
   *
   * @param layerRows The number of rows of neurons.
   * @param layerCols The number of columns of neurons.
   */
  SoftmaxLayer(const size_t layerRows, const size_t layerCols) :
      inputActivations(arma::zeros<DataType>(layerRows, layerCols)),
      delta(arma::zeros<DataType>(layerRows, layerCols)),
      layerRows(layerRows),
      layerCols(layerCols),
      layerSlices(1),
      outputMaps(1)
  {
    // Nothing to do here.
  }

  /**
   * Create n-dimensional SoftmaxLayer object using the specified rows and
   * columns and number of slices. In this case, DataType must be arma::cube.
   *
   * @param layerRows The number of rows of neurons.
   * @param layerCols The number of columns of neurons.
   * @param layerCols The number of slices of neurons.
   * @param layerCols The number of output maps.
   */
  SoftmaxLayer(const size_t layerRows,
               const size_t layerCols,
               const size_t layerSlices,
               const size_t outputMaps = 1) :
      inputActivations(arma::zeros<DataType>(layerRows, layerCols,
          layerSlices * outputMaps)),
      delta(arma::zeros<DataType>(layerRows, layerCols,
          layerSlices * outputMaps)),
      layerRows(layerRows),
      layerCols(layerCols),
      layerSlices(layerSlices),
      outputMaps(outputMaps)
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
  void FeedForward(const DataType& inputActivation, DataType& outputActivation)
  {
    outputActivation = arma::trunc_exp(inputActivation -
        arma::repmat(arma::max(inputActivation), inputActivation.n_rows, 1));
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
  void FeedBackward(const DataType& /* unused */,
                    const DataType& error,
                    DataType& delta)
  {
    delta = error;
  }

  //! Get the input activations.
  DataType& InputActivation() const { return inputActivations; }
  //! Modify the input activations.
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

  //! Get the number of output maps.
  size_t OutputMaps() const { return outputMaps; }

  //! The the value of the deterministic parameter.
  bool Deterministic() const {return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() {return deterministic; }

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

  //! Locally-stored number of output maps.
  size_t outputMaps;

  //! Locally-stored deterministic parameter.
  bool deterministic;
}; // class SoftmaxLayer

}; // namespace ann
}; // namespace mlpack

#endif
