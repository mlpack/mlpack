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
    typename OptimizerType,
    class WeightInitRule = NguyenWidrowInitialization<>,
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
      inputLayer(inputLayer), outputLayer(outputLayer), optimizer(optimizer)
  {
    weightInitRule.Initialize(weights, outputLayer.InputSize(),
        inputLayer.OutputSize());
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified activity function.
   */
  void FeedForward(const VecType& input)
  {
    outputLayer.InputActivation() += (weights * input);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param error The backpropagated error.
   */
  void FeedBackward(const VecType& error)
  {
    // Calculating the delta using the partial derivative of the error with
    // respect to a weight.
    delta = (weights.t() * error);
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
  OptimizerType& Optimzer() const { return optimizer; }
  //! Modify the optimzer.
  OptimizerType& Optimzer() { return optimizer; }

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

  //! Locally-stored optimzer object.
  OptimizerType& optimizer;

  //! Locally-stored detla object that holds the calculated delta.
  VecType delta;
}; // class FullConnection

}; // namespace ann
}; // namespace mlpack

#endif
