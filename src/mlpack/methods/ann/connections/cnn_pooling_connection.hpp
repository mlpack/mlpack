/**
 * @file cnn_pooling_connection.hpp
 * @author Shangtong Zhang
 *
 * Implementation of the pooling connection between input layer
 * and output layer for CNN.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_POOLING_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_POOLING_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/pooling/max_pooling.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

namespace mlpack{
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the pooling connection class for CNN.
 * The pooling connection connects
 * input layer with the output layer by pooling.
 * output = factor * pooling_value + bias
 *
 * @tparam InputLayerType Type of the connected input layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam PoolingRule Type of pooling strategy.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename OptimizerType,
    typename PoolingRule = MaxPooling,
    typename MatType = arma::mat
>
class PoolingConnection
{
 public:
  /**
   * Create the PoolingConnection object using the specified input layer, output
   * layer, optimizer, factor, bias and pooling strategy.
   * The factor and bias is stored in @weights.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param OptimizerType The optimizer used to update the weight matrix.
   * @param PoolingRule The strategy of pooling.
   */
  PoolingConnection(InputLayerType& inputLayer,
                    OutputLayerType& outputLayer,
                    OptimizerType& optimizer,
                    double factor = 1.0,
                    double bias = 0,
                    PoolingRule pooling = PoolingRule()) :
      inputLayer(inputLayer), outputLayer(outputLayer), optimizer(optimizer),
      weights(2), pooling(pooling),
      rawOutput(outputLayer.InputActivation().n_rows,
                outputLayer.InputActivation().n_cols)
  {
    delta = arma::zeros<MatType>(inputLayer.InputActivation().n_rows,
                                 inputLayer.InputActivation().n_cols);
    weightsDelta = arma::zeros<arma::colvec>(2);
    weights(0) = factor;
    weights(1) = bias;
  }
  
  /**
   * Ordinary feed forward pass of a neural network, 
   * apply pooling to the neurons in the input layer.
   *
   * @param input Input data used for pooling.
   */
  void FeedForward(const MatType& input)
  {
    Log::Debug << "PoolingConnection::FeedForward" << std::endl;
    Log::Debug << "Input:\n" << input << std::endl;
    Log::Debug << "Weights:\n" << weights << std::endl;
    size_t r_step = input.n_rows / outputLayer.InputActivation().n_rows;
    size_t c_step = input.n_cols / outputLayer.InputActivation().n_cols;
    for (size_t j = 0; j < input.n_cols; j += c_step)
    {
      for (size_t i = 0; i < input.n_rows; i += r_step)
      {
        double value = 0;
        pooling.pooling(input(arma::span(i, i + r_step -1),
                              arma::span(j, j + c_step - 1)), value);
        rawOutput(i / r_step, j / c_step) = value;
      }
    }
    outputLayer.InputActivation() += rawOutput * weights(0) + weights(1);
  }
  
  /**
   * Ordinary feed backward pass of a neural network.
   * Apply unsampling to the error in output layer to 
   * pass the error to input layer.
   * @param error The backpropagated error.
   */
  void FeedBackward(const MatType& error)
  {
    Log::Debug << "PoolingConnection::FeedBackward" << std::endl;
    Log::Debug << "Error:\n" << error << std::endl;
    weightsDelta(1) = arma::sum(arma::sum(error));
    weightsDelta(0) = arma::sum(arma::sum(rawOutput % error));
    MatType weightedError = error * weights(0);
    size_t r_step = inputLayer.InputActivation().n_rows / error.n_rows;
    size_t c_step = inputLayer.InputActivation().n_cols / error.n_cols;
    const MatType& input = inputLayer.InputActivation();
    MatType newError;
    for (size_t j = 0; j < input.n_cols; j += c_step)
    {
      for (size_t i = 0; i < input.n_rows; i += r_step)
      {
        const MatType& inputArea = input(arma::span(i, i + r_step -1),
                                         arma::span(j, j + c_step - 1));
        pooling.unpooling(inputArea,
                          weightedError(i / r_step, j / c_step),
                          newError);
        delta(arma::span(i, i + r_step -1),
              arma::span(j, j + c_step - 1)) = newError;
      }
    }
    Log::Debug << "Delta:\n" << delta << std::endl;
    Log::Debug << "WeightsDelta:\n" << weightsDelta << std::endl;
    inputLayer.Delta() += delta;
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
  
  //! Get the optimizer.
  OptimizerType& Optimzer() const { return optimizer; }
  //! Modify the optimzer.
  OptimizerType& Optimzer() { return optimizer; }
  
  //! Get the passed error in backward propagation.
  MatType& Delta() const { return delta; }
  //! Modify the passed error in backward propagation.
  MatType& Delta() { return delta; }
  
  //! Get the detla of weights.
  MatType& WeightsDelta() const { return weightsDelta; }
  //! Modify the delta of weights.
  MatType& WeightsDelta() { return weightsDelta; }
  
  //! Get the pooling strategy.
  PoolingRule& Pooling() const { return pooling; }
  //! Modify the pooling strategy.
  PoolingRule& Pooling() { return pooling; }
  
 private:
  //! Locally-stored input layer.
  InputLayerType& inputLayer;
  
  //! Locally-stored output layer.
  OutputLayerType& outputLayer;
  
  //! Locally-stored optimizer.
  OptimizerType& optimizer;
  
  //! Locally-stored weights, only two value, factor and bias.
  arma::colvec weights;
  
  //! Locally-stored passed error in backward propagation.
  MatType delta;
  
  //! Locally-stored pooling strategy.
  PoolingRule pooling;
  
  //! Locally-stored delta of weights.
  MatType weightsDelta;
  
  /**
   * Locally-stored raw result of pooling,
   * before multiplied by factor and added by bias.
   * Cache it to speed up when performing backward propagation.
   */
  MatType rawOutput;
};

template<
    typename InputLayerType,
    typename OutputLayerType,
    typename OptimizerType,
    typename PoolingRule,
    typename MatType >
class ConnectionTraits<
    PoolingConnection<InputLayerType, OutputLayerType, OptimizerType,
    PoolingRule, MatType> > {
 public:
  static const bool IsSelfConnection = false;
  static const bool IsFullselfConnection = false;
  static const bool hasWeightsDelta = true;
};// class PoolingConnections
  
}; // namespace ann
}; // namespace mlpack

#endif