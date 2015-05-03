/**
 * @file full_connection.hpp
 * @author Marcus Edel
 *
 * Implementation of the full connection class.
 */
#ifndef __MLPACK_METHOS_ANN_CONNECTIONS_FULL_CONNECTION_HPP
#define __MLPACK_METHOS_ANN_CONNECTIONS_FULL_CONNECTION_HPP

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
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename OptimizerType = SteepestDescent<>,
    class WeightInitRule = NguyenWidrowInitialization,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class FullConnection
{
 public:
  /**
   * Create the FullConnection object using the specified input layer, output
   * layer, optimizer and weight initialize rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param OptimizerType The optimizer used to update the weight matrix.
   * @param WeightInitRule The weight initialize rule used to initialize the
   * weight matrix.
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
        inputLayer.OutputSize() * inputLayer.LayerSlices());
  }

  FullConnection(InputLayerType& inputLayer,
               OutputLayerType& outputLayer,
               WeightInitRule weightInitRule = WeightInitRule()) :
    inputLayer(inputLayer),
    outputLayer(outputLayer),
    optimizer(new OptimizerType()),
    ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outputLayer.InputSize(),
        inputLayer.OutputSize() * inputLayer.LayerSlices());
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
    // Vectorise the input (cube of n slices with a 1x1 dense matrix) and
    // perform the feed forward pass.
    outputLayer.InputActivation() += (weights *
        arma::vec(input.memptr(), input.n_slices));
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param error The backpropagated error.
   */
  template<typename eT>
  void FeedBackward(const arma::Col<eT>& error)
  {
    // Calculating the delta using the partial derivative of the error with
    // respect to a weight.
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
    gradient = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);

    // Vectorise the input (cube of n slices with a 1x1 dense matrix) and
    // calculate the gradient.
    gradient.slice(0) = outputLayer.Delta() *
        arma::rowvec(inputLayer.InputActivation().memptr(),
        inputLayer.InputActivation().n_elem);
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
  VecType& Delta() const { return delta; }
 //  //! Modify the delta.
  VecType& Delta() { return delta; }

 private:
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
  VecType delta;
}; // class FullConnection

}; // namespace ann
}; // namespace mlpack

#endif
