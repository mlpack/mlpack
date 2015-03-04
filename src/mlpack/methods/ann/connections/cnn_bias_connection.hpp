/**
 * @file cnn_bias_connection.hpp
 * @author Shangtong Zhang
 *
 * Implementation of the connection between bias layer and other layer.
 */
#ifndef __MLPACK_METHOS_ANN_CONNECTIONS_BIAS_CONNECTION_HPP
#define __MLPACK_METHOS_ANN_CONNECTIONS_BIAS_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the bias connection class. The bias connection connects
 * bias layer and other layer for a convolutional neural network. In a typical 
 * CNN, all neuron in a layer always share a common bias, so actually this class
 * is equal to a double value, this value is stored in @weights.
 *
 * @tparam InputLayerType Type of the connected input layer. 
 *         It must be a bias layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weights matrix.
 *         Acutally this matrix only has one value.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename OptimizerType,
    class WeightInitRule = NguyenWidrowInitialization<>,
    typename MatType = arma::mat
>
class BiasConnection
{
 public:
  /**
   * Create the BiasConnection object using the specified input layer, output
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
  BiasConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 OptimizerType& optimizer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
  inputLayer(inputLayer), outputLayer(outputLayer), optimizer(optimizer) {
    if (!LayerTraits<typename std::remove_reference<decltype(
        inputLayer)>::type>::IsBiasLayer) {
    // Input layer must be bias layer.
      Log::Fatal << "Input layer isn't bias layer!" << std::endl;
    }
    if (inputLayer.OutputSize() != 1) {
    /** 
     * A typical bias layer only has one neuron.
     * The value of bias is actually stored in @weights.
     * The bias layer is useless actually.
     */
      Log::Fatal << "The size of bias layer must be 1 !" << std::endl;
    }
    weightInitRule.Initialize(weights, 1, 1);
    weightsDelta = arma::zeros<MatType>(1, 1);
    delta = arma::zeros<MatType>(1, 1);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   */
  void FeedForward(const MatType& /* used */)
  {
    Log::Debug << "BiasConnection::FeedForward" << std::endl;
    Log::Debug << "Weights:\n" << weights << std::endl;
    
    // All neuron share a common bias.
    outputLayer.InputActivation() += weights(0, 0);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param error The backpropagated error.
   */
  void FeedBackward(const MatType& error)
  {
    // Calculate the delta of weights to update @weights,
    // which is actually the bias value.
    weightsDelta = arma::accu(error);
  }
  
  //! Get the optimzer.
  OptimizerType& Optimzer() const { return optimizer; }
  //! Modify the optimzer.
  OptimizerType& Optimzer() { return optimizer; }
  
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

  //! Get the detla of weights.
  MatType& WeightsDelta() const { return weightsDelta; }
  //! Modify the delta of weights.
  MatType& WeightsDelta() { return weightsDelta; }
  
  //! Get the passed error.
  MatType& Delta() const { return delta; }
  //! Modify the passed error.
  MatType& Delta() { return delta; }

 private:

  //! Locally-stored connected input layer object.
  InputLayerType& inputLayer;

  //! Locally-stored connected output layer object.
  OutputLayerType& outputLayer;
  
  //! Locally-stored optimizer.
  OptimizerType& optimizer;
  
  //! Locally-stored delta of weights
  MatType weightsDelta;
  
  //! Locally-stored weights.
  MatType weights;
  
  //! Locally-stored passed error in backward propagation.
  MatType delta;
}; // class BiasConnection
  
template<
  typename InputLayerType,
  typename OutputLayerType,
  typename OptimizerType,
  class WeightInitRule,
  typename MatType>
class ConnectionTraits<
  BiasConnection<InputLayerType, OutputLayerType, OptimizerType,
                 WeightInitRule, MatType> > {
    
 public:
  static const bool IsSelfConnection = false;
  static const bool IsFullselfConnection = false;
  static const bool hasWeightsDelta = true;
    
};

}; // namespace ann
}; // namespace mlpack

#endif
