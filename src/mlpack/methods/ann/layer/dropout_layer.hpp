/**
 * @file dropout_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the DropoutLayer class, which implements a regularizer that
 * randomly sets units to zero. This prevents units from co-adapting too much.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_DROPOUT_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_DROPOUT_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The dropout layer is a regularizer that randomly with probability ratio
 * sets input values to zero and scales the remaining elements by factor 1 /
 * (1 - ratio). If rescale is true the input is scaled with 1 / (1-p) when
 * deterministic is false. In the deterministic mode (during testing), the layer
 * just scales the output.
 *
 * Note: During training you should set deterministic to false and during
 * testing you should set deterministic to true.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Hinton2012,
 *   author  = {Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky,
 *              Ilya Sutskever, Ruslan Salakhutdinov},
 *   title   = {Improving neural networks by preventing co-adaptation of feature
 *              detectors},
 *   journal = {CoRR},
 *   volume  = {abs/1207.0580},
 *   year    = {2012},
 * }
 * @endcode
 *
 * @tparam DataType Type of data (arma::colvec, arma::mat arma::sp_mat or
 *    arma::cube).
 */
template <
    typename DataType = arma::colvec
>
class DropoutLayer
{
 public:
  /**
   * Create the DropoutLayer object using the specified parameter.
   *
   * @param layerSize The number of neurons.
   * @param ratio The probability of setting a value to zero.
   * @param rescale If true the input is rescaled when deterministic is False.
   */
  DropoutLayer(const size_t layerSize,
               const double ratio = 0.5,
               const bool rescale = true) :
      inputActivations(arma::zeros<DataType>(layerSize)),
      delta(arma::zeros<DataType>(layerSize)),
      layerRows(layerSize),
      layerCols(1),
      layerSlices(1),
      outputMaps(1),
      ratio(ratio),
      rescale(rescale)
  {
    // Nothing to do here.
  }

  /**
   * Create 2-dimensional DropoutLayer object using the specified rows and
   * columns. In this case, DataType must be arma::mat or arma::sp_mat.
   *
   * @param layerRows The number of rows of neurons.
   * @param layerCols The number of columns of neurons.
   * @param ratio The probability of setting a value to zero.
   * @param rescale If true the input is rescaled when deterministic is False.
   */
  DropoutLayer(const size_t layerRows,
               const size_t layerCols,
               const double ratio = 0.5,
               const bool rescale = true) :
      inputActivations(arma::zeros<DataType>(layerRows, layerCols)),
      delta(arma::zeros<DataType>(layerRows, layerCols)),
      layerRows(layerRows),
      layerCols(layerCols),
      layerSlices(1),
      outputMaps(1),
      ratio(ratio),
      rescale(rescale)
  {
    // Nothing to do here.
  }

  /**
   * Create n-dimensional DropoutLayer object using the specified rows and
   * columns and number of slices. In this case, DataType must be arma::cube.
   *
   * @param layerRows The number of rows of neurons.
   * @param layerCols The number of columns of neurons.
   * @param layerCols The number of slices of neurons.
   * @param layerCols The number of output maps.
   * @param ratio The probability of setting a value to zero.
   * @param rescale If true the input is rescaled when deterministic is False.
   */
  DropoutLayer(const size_t layerRows,
               const size_t layerCols,
               const size_t layerSlices,
               const size_t outputMaps = 1,
               const double ratio = 0.5,
               const bool rescale = true) :
      inputActivations(arma::zeros<DataType>(layerRows, layerCols,
          layerSlices * outputMaps)),
      delta(arma::zeros<DataType>(layerRows, layerCols,
          layerSlices * outputMaps)),
      layerRows(layerRows),
      layerCols(layerCols),
      layerSlices(layerSlices),
      outputMaps(outputMaps),
      ratio(ratio),
      rescale(rescale)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of the dropout layer.
   *
   * @param inputActivation Input data used for evaluating the dropout layer.
   * @param outputActivation Data to store the resulting output activation.
   */
  template<typename eT>
  void FeedForward(const arma::Mat<eT>& inputActivation,
                   arma::Mat<eT>& outputActivation)
  {
    // The dropout mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
      outputActivation = inputActivation;

      if (rescale)
        outputActivation *= scale;
    }
    else
    {
      // Scale with input / (1 - ratio) and set values to zero with probability
      // ratio.
      scale = 1.0 / (1.0 - ratio);
      mask = arma::randu<arma::Mat<eT> >(layerRows, layerCols);
      mask.transform( [&](double val) { return val > ratio; } );
      outputActivation = inputActivation % mask * scale;
    }
  }

  /**
   * Ordinary feed forward pass of the dropout layer.
   *
   * @param inputActivation Input data used for evaluating the dropout layer.
   * @param outputActivation Data to store the resulting output activation.
   */
  template<typename eT>
  void FeedForward(const arma::Cube<eT>& inputActivation,
                   arma::Cube<eT>& outputActivation)
  {
    // The dropout mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
      outputActivation = inputActivation;

      if (rescale)
        outputActivation *= scale;
    }
    else
    {
      // Scale with input / (1 - ratio) and set values to zero with probability
      // ratio.
      scale = 1.0 / (1.0 - ratio);
      mask = arma::randu<arma::Cube<eT> >(layerRows, layerCols,
          layerSlices * outputMaps);
      mask.transform( [&](double val) { return (val > ratio); } );
      outputActivation = inputActivation % mask * scale;
    }
  }

  /**
   * Ordinary feed backward pass of the dropout layer.
   *
   * @param error The backpropagated error.
   * @param delta The calculating delta using the delta from the previous layer.
   */
  void FeedBackward(const DataType& /* unused */,
                    const DataType& error,
                    DataType& delta)
  {
    delta = error % mask * scale;
  }

  /**
   * Ordinary feed backward pass of the dropout layer.
   *
   * @param inputActivation Input data used to map the error from the previous
   *    layer.
   * @param error The backpropagated error.
   * @param delta The calculating delta using the delta from the previous layer.
   */
  template<typename eT>
  void FeedBackward(const arma::Cube<eT>& inputActivation,
                    const arma::Mat<eT>& error,
                    arma::Cube<eT>& delta)
  {
    delta = delta % mask * scale;

    // Generate a cube from the error matrix.
    arma::Cube<eT> mappedError = arma::zeros<arma::cube>(inputActivation.n_rows,
        inputActivation.n_cols, inputActivation.n_slices);

    for (size_t s = 0, j = 0; s < mappedError.n_slices; s+= error.n_cols, j++)
    {
      for (size_t i = 0; i < error.n_cols; i++)
      {
        arma::Col<eT> temp = error.col(i).subvec(
            j * inputActivation.n_rows * inputActivation.n_cols,
            (j + 1) * inputActivation.n_rows * inputActivation.n_cols - 1);

        mappedError.slice(s + i) = arma::Mat<eT>(temp.memptr(),
            inputActivation.n_rows, inputActivation.n_cols);
      }
    }

    delta = mappedError;
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

  //! The value of the deterministic parameter.
  bool Deterministic() const {return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() {return deterministic; }

  //! The probability of setting a value to zero.
  double Ratio() const {return ratio; }
  //! Modify the probability of setting a value to zero.
  double& Ratio() {return ratio; }

  //! The value of the rescale parameter.
  bool Rescale() const {return rescale; }
  //! Modify the value of the rescale parameter.
  bool& Rescale() {return rescale; }

 private:
  //! Locally-stored input activation object.
  DataType inputActivations;

  //! Locally-stored delta object.
  DataType delta;

  //! Locally-stored mast object.
  DataType mask;

  //! Locally-stored number of layer rows.
  size_t layerRows;

  //! Locally-stored number of layer cols.
  size_t layerCols;

  //! Locally-stored number of layer slices.
  size_t layerSlices;

  //! Locally-stored number of output maps.
  size_t outputMaps;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! If true the input is rescaled when deterministic is False.
  bool rescale;
}; // class DropoutLayer

}; // namespace ann
}; // namespace mlpack

#endif
