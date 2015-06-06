/**
 * @file full_connection.hpp
 * @author Marcus Edel
 *
 * Implementation of the full connection class.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_FULL_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_FULL_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/optimizer/steepest_descent.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the full connection class. The full connection connects
 * every neuron from the input layer with the output layer in a matrix
 * multiplicative way.
 *
 * @tparam InputLayerType Type of the connected input layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weight matrix.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename OptimizerType = SteepestDescent<>,
    class WeightInitRule = NguyenWidrowInitialization,
    typename MatType = arma::mat
>
class FullConnection
{
 public:
  /**
   * Create the FullConnection object using the specified input layer, output
   * layer, optimizer and weight initialization rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param OptimizerType The optimizer used to update the weight matrix.
   * @param WeightInitRule The weights initialization rule used to initialize the
   * weights matrix.
   */
  FullConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 OptimizerType& optimizer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(&optimizer),
      ownsOptimizer(false)
  {
    weightInitRule.Initialize(weights, outputLayer.InputSize(),
        inputLayer.LayerRows() * inputLayer.LayerCols() *
        inputLayer.LayerSlices() * inputLayer.OutputMaps() /
        outputLayer.LayerCols());
  }

  /**
   * Create the FullConnection object using the specified input layer, output
   * layer and weight initialization rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param WeightInitRule The weights initialization rule used to initialize the
   * weights matrix.
   */
  FullConnection(InputLayerType& inputLayer,
               OutputLayerType& outputLayer,
               WeightInitRule weightInitRule = WeightInitRule()) :
    inputLayer(inputLayer),
    outputLayer(outputLayer),
    optimizer(new OptimizerType(outputLayer.InputSize(), inputLayer.LayerRows()
        * inputLayer.LayerCols() * inputLayer.LayerSlices() *
        inputLayer.OutputMaps() / outputLayer.LayerCols())),
    ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outputLayer.InputSize(),
        inputLayer.LayerRows() * inputLayer.LayerCols() *
        inputLayer.LayerSlices() * inputLayer.OutputMaps() /
        outputLayer.LayerCols());
  }

  /**
   * Delete the full connection object and its optimizer.
   */
  ~FullConnection()
  {
    if (ownsOptimizer)
      delete optimizer;
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f using a dense matrix as
   * input.
   *
   * @param input Input data used for evaluating the specified activity function.
   */
  template<typename eT>
  void FeedForward(const arma::Mat<eT>& input)
  {
    outputLayer.InputActivation() += (weights * input);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f using a 3rd order tensor
   * as input.
   *
   * @param input Input data used for evaluating the specified activity function.
   */
  template<typename eT>
  void FeedForward(const arma::Cube<eT>& input)
  {
    MatType data(input.n_elem / outputLayer.LayerCols(),
        outputLayer.LayerCols());

    for (size_t s = 0, c = 0; s < input.n_slices / data.n_cols; s++)
    {
      for (size_t i = 0; i < data.n_cols; i++, c++)
      {
        data.col(i).subvec(s * input.n_rows * input.n_cols, (s + 1) *
            input.n_rows * input.n_cols - 1) = arma::vectorise(input.slice(c));
      }
    }

    outputLayer.InputActivation() += (weights * data);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param error The backpropagated error.
   */
  template<typename ErrorType>
  void FeedBackward(const ErrorType& error)
  {
    delta = (weights.t() * error);
  }

  /*
   * Calculate the gradient using the output delta (dense matrix) and the input
   * activation (dense matrix).
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>& gradient)
  {
    gradient = outputLayer.Delta() * inputLayer.InputActivation().t();
  }

  /*
   * Calculate the gradient using the output delta (3rd oder tensor) and the
   * input activation (3rd oder tensor).
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Cube<eT>& gradient)
  {
     GradientDelta(outputLayer.Delta(), gradient);
  }

  //! Get the weights.
  MatType& Weights() const { return weights; }
  //! Modify the weights.
  MatType& Weights() { return weights; }

  //! Get the input layer.
  InputLayerType& InputLayer() const { return inputLayer; }
  //! Modify the input layer.
  InputLayerType& InputLayer() { return inputLayer; }

  //! Get the output layer.
  OutputLayerType& OutputLayer() const { return outputLayer; }
  //! Modify the output layer.
  OutputLayerType& OutputLayer() { return outputLayer; }

  //! Get the optimzer.
  OptimizerType& Optimzer() const { return *optimizer; }
  //! Modify the optimzer.
  OptimizerType& Optimzer() { return *optimizer; }

  //! Get the detla.
  MatType& Delta() const { return delta; }
 //  //! Modify the delta.
  MatType& Delta() { return delta; }

 private:
   /*
   * Calculate the gradient using the output delta (3rd oder tensor) and the
   * input activation (3rd oder tensor).
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(arma::Mat<eT>& delta, arma::Cube<eT>& gradient)
  {
    gradient = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    arma::Mat<eT> data = arma::Mat<eT>(outputLayer.Delta().n_cols,
        inputLayer.InputActivation().n_elem / outputLayer.Delta().n_cols);

    for (size_t s = 0, c = 0; s < inputLayer.InputActivation().n_slices /
        data.n_rows; s++)
    {
      for (size_t i = 0; i < data.n_rows; i++, c++)
      {
        data.row(i).subvec(s * inputLayer.InputActivation().n_rows *
            inputLayer.InputActivation().n_cols, (s + 1) *
            inputLayer.InputActivation().n_rows *
            inputLayer.InputActivation().n_cols - 1) = arma::vectorise(
                inputLayer.InputActivation().slice(c), 1);
      }
    }

    gradient.slice(0) = outputLayer.Delta() * data / outputLayer.Delta().n_cols;
  }

  /*
   * Calculate the gradient using the output delta (3rd oder tensor) and the
   * input activation (3rd oder tensor).
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(arma::Cube<eT>& delta, arma::Cube<eT>& gradient)
  {
    gradient = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
  }

  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored connected input layer object.
  InputLayerType& inputLayer;

  //! Locally-stored connected output layer object.
  OutputLayerType& outputLayer;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;

  //! Locally-stored detla object that holds the calculated delta.
  MatType delta;
}; // class FullConnection

}; // namespace ann
}; // namespace mlpack

#endif
