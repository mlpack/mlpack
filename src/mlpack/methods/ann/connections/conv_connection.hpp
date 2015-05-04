/**
 * @file cnn_conv_connection.hpp
 * @author Shangtong Zhang
 *
 * Implementation of the convolutional connection 
 * between input layer and output layer.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_CONV_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_CONV_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/convolution/valid_convolution.hpp>
#include <mlpack/methods/ann/convolution/full_convolution.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

namespace mlpack{
namespace ann  /** Artificial Neural Network. */{
/**
 * Implementation of the convolutional connection class.
 * The convolutional connection performs convolution between input layer
 * and output layer.
 * Convolution is applied to every neuron in input layer.
 * The kernel used for convolution is stored in @weights.
 *
 * Users can design their own convolution rule (ForwardConvolutionRule)
 * to perform forward process. But once user-defined forward convolution is used, 
 * users have to design special BackwardConvolutionRule and GradientConvolutionRule
 * to perform backward process and calculate gradient corresponding to 
 * ForwardConvolutionRule, aimed to guarantee the correctness of error flow.
 *
 * @tparam InputLayerType Type of the connected input layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weights matrix.
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename OptimizerType,
    class WeightInitRule = NguyenWidrowInitialization<>,
    typename ForwardConvolutionRule = ValidConvolution,
    typename BackwardConvolutionRule = ValidConvolution,
    typename GradientConvolutionRule = RotatedKernelFullConvolution,
    typename MatType = arma::mat
>
class ConvConnection
{
 public:
  /**
   * Create the ConvConnection object using the specified input layer, output
   * layer, optimizer and weight initialize rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param OptimizerType The optimizer used to update the weights matrix.
   * @param weightsRows The number of rows of convolutional kernel.
   * @param weightsCols The number of cols of convolutional kernel.
   * @param WeightInitRule The weights initialize rule used to initialize the
   * weights matrix.
   */
  ConvConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 OptimizerType& optimizer,
                 size_t weightsRows,
                 size_t weightsCols,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer), outputLayer(outputLayer), optimizer(optimizer)
  {
    weightInitRule.Initialize(weights, weightsRows, weightsCols);
    gradient = arma::zeros<MatType>(weightsRows, weightsCols);
  }
  
  /**
   * Ordinary feed forward pass of a neural network, 
   * Apply convolution to every neuron in input layer and
   * put the output in the output layer.
   */
  void FeedForward(const MatType& input)
  {
    MatType output(outputLayer.InputActivation().n_rows,
                   outputLayer.InputActivation().n_cols);
    ForwardConvolutionRule::conv(input, weights, output);
    outputLayer.InputActivation() += output;
  }
  
  /**
   * Ordinary feed backward pass of a neural network.
   * Pass the error from output layer to input layer and 
   * calculate the delta of kernel weights.
   * @param error The backpropagated error.
   */
  void FeedBackward(const MatType& error)
  {
    BackwardConvolutionRule::conv(inputLayer.InputActivation(), error, gradient);
    GradientConvolutionRule::conv(weights, error, delta);
    inputLayer.Delta() += delta;
  }
  
  //! Get the convolution kernel.
  MatType& Weights() const { return weights; }
  //! Modify the convolution kernel.
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
  
  //! Get the passed error in backward propagation.
  MatType& Delta() const { return delta; }
  //! Modify the passed error in backward propagation.
  MatType& Delta() { return delta; }
  
  //! Get the gradient of kernel.
  MatType& Gradient() const { return gradient; }
  //! Modify the gradient of kernel.
  MatType& Gradient() { return gradient; }
  
 private:
  //! Locally-stored kernel weights.
  MatType weights;
  
  //! Locally-stored inputlayer.
  InputLayerType& inputLayer;
  
  //! Locally-stored outputlayer.
  OutputLayerType& outputLayer;
  
  //! Locally-stored optimizer.
  OptimizerType& optimizer;
  
  //! Locally-stored passed error in backward propagation.
  MatType delta;
  
  //! Locally-stored gradient of kernel weights.
  MatType gradient;
};// class ConvConnection
    
}; // namespace ann
}; // namespace mlpack

#endif