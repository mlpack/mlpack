/**
 * @file self_connection.hpp
 * @author Marcus Edel
 *
 * Implementation of the self connection class. This connection is mainly used
 * as recurrent connection.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_SELF_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_SELF_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the self connection class. The self connection connects
 * every neuron from the input layer with the output layer in a multiplicative
 * way, except the elements on the main diagonal.
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
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = NguyenWidrowInitialization,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class SelfConnection
{
 public:
  /**
   * Create the SelfConnection object using the specified input layer, output
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
  SelfConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 OptimizerType<SelfConnection<InputLayerType,
                                              OutputLayerType,
                                              OptimizerType,
                                              WeightInitRule,
                                              MatType,
                                              VecType>, MatType>& optimizer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(&optimizer),
      ownsOptimizer(false),
      connection(1 - arma::eye<MatType>(inputLayer.OutputSize(),
          inputLayer.OutputSize()))
  {
    weightInitRule.Initialize(weights, outputLayer.InputSize(),
        inputLayer.OutputSize());
  }

  /**
   * Create the SelfConnection object using the specified input layer, output
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
  SelfConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(new OptimizerType<SelfConnection<InputLayerType,
                                                 OutputLayerType,
                                                 OptimizerType,
                                                 WeightInitRule,
                                                 MatType,
                                                 VecType>, MatType>(*this)),
      ownsOptimizer(true),
      connection(1 - arma::eye<MatType>(inputLayer.OutputSize(),
          inputLayer.OutputSize()))
  {
    weightInitRule.Initialize(weights, outputLayer.InputSize(),
        inputLayer.OutputSize());
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified activity
   * function.
   */
  void FeedForward(const VecType& input)
  {
    outputLayer.InputActivation() += (weights % connection) * input;
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
    delta = (weights % connection).t() * error;
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param gradient The calculated gradient.
   */
  void Gradient(MatType& gradient)
  {
    gradient = outputLayer.Delta() * inputLayer.InputActivation().t();
  }

  //! Get the weights.
  const MatType& Weights() const { return weights; }
  //! Modify the weights.
  MatType& Weights() { return weights; }

  //! Get the input layer.
  const InputLayerType& InputLayer() const { return inputLayer; }
  //! Modify the input layer.
  InputLayerType& InputLayer() { return inputLayer; }

  //! Get the output layer.
  const OutputLayerType& OutputLayer() const { return outputLayer; }
  //! Modify the output layer.
  OutputLayerType& OutputLayer() { return outputLayer; }

    //! Get the optimizer.
  OptimizerType<SelfConnection<InputLayerType,
                               OutputLayerType,
                               OptimizerType,
                               WeightInitRule,
                               MatType,
                               VecType>, MatType>& Optimizer() const
  {
    return *optimizer;
  }
  //! Modify the optimizer.
  OptimizerType<SelfConnection<InputLayerType,
                               OutputLayerType,
                               OptimizerType,
                               WeightInitRule,
                               MatType,
                               VecType>, MatType>& Optimizer()
  {
    return *optimizer;
  }

  //! Get the detla.
  const VecType& Delta() const { return delta; }
  //! Modify the delta.
  VecType& Delta() { return delta; }

 private:
  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored connected input layer object.
  InputLayerType& inputLayer;

  //! Locally-stored connected output layer object.
  OutputLayerType& outputLayer;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<SelfConnection<InputLayerType,
                               OutputLayerType,
                               OptimizerType,
                               WeightInitRule,
                               MatType,
                               VecType>, MatType>* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;

  //! Locally-stored detla object that holds the calculated delta.
  VecType delta;

  //! Locally-stored connection multiplication type.
  MatType connection;
}; // class SelfConnection

//! Connection traits for the self connection.
template<
    typename InputLayerType,
    typename OutputLayerType,
    template<typename, typename> class OptimizerType,
    class WeightInitRule,
    typename MatType,
    typename VecType
>
class ConnectionTraits<
    SelfConnection<InputLayerType, OutputLayerType, OptimizerType,
    WeightInitRule, MatType, VecType> >
{
 public:
  static const bool IsSelfConnection = true;
  static const bool IsFullselfConnection = false;
  static const bool IsPoolingConnection = false;
  static const bool IsIdentityConnection = false;
};

}; // namespace ann
}; // namespace mlpack

#endif
