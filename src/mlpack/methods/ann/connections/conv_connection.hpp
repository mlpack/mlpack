/**
 * @file cnn_conv_connection.hpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Implementation of the convolutional connection between input layer and output
 * layer.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_CONV_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_CONV_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/optimizer/steepest_descent.hpp>
#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>

namespace mlpack{
namespace ann  /** Artificial Neural Network. */{

/**
 * Implementation of the convolutional connection class. The convolutional
 * connection performs convolution between input layer and output layer.
 * Convolution is applied to every neuron in input layer. The kernel used for
 * convolution is stored in @weights.
 *
 * Users can design their own convolution rule (ForwardConvolutionRule)
 * to perform forward process. But once user-defined forward convolution is
 * used, users have to design special BackwardConvolutionRule and
 * GradientConvolutionRule to perform backward process and calculate gradient
 * corresponding to ForwardConvolutionRule, aimed to guarantee the correctness
 * of error flow.
 *
 * @tparam InputLayerType Type of the connected input layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weights matrix.
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam DataType Type of data (arma::mat, arma::sp_mat or arma::cube).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename OptimizerType = SteepestDescent<>,
    class WeightInitRule = RandomInitialization,
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename DataType = arma::cube
>
class ConvConnection
{
 public:
  /**
   * Create the ConvConnection object using the specified input layer, output
   * layer, optimizer and weight initialization rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param filterRows The number of rows of the convolutional kernel.
   * @param filterCols The number of cols of the convolutional kernel.
   * @param OptimizerType The optimizer used to update the weight matrix.
   * @param WeightInitRule The weights initialization rule used to initialize
   * the weights matrix.
   */
  ConvConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 const size_t filterRows,
                 const size_t filterCols,
                 OptimizerType& optimizer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(&optimizer),
      ownsOptimizer(false)
  {
    weightInitRule.Initialize(weights, filterRows, filterCols,
        outputLayer.LayerSlices());
  }

  /**
   * Create the ConvConnection object using the specified input layer, output
   * layer, optimizer and weight initialization rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param filterRows The number of rows of the convolutional kernel.
   * @param filterCols The number of cols of the convolutional kernel.
   * @param WeightInitRule The weights initialization rule used to initialize
   * the weights matrix.
   */
  ConvConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 const size_t filterRows,
                 const size_t filterCols,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(new OptimizerType()),
      ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, filterRows, filterCols,
        outputLayer.LayerSlices());
  }

  /**
   * Delete the conv connection object and its optimizer.
   */
  ~ConvConnection()
  {
    if (ownsOptimizer)
      delete optimizer;
  }

  /**
   * Ordinary feed forward pass of a neural network. Apply convolution to every
   * neuron in input layer and put the output in the output layer.
   */
  template<typename InputType>
  void FeedForward(const InputType& input)
  {
    DataType output;
    ForwardConvolutionRule::Convolution(input, weights, output);
    outputLayer.InputActivation() += output;
  }

  /**
   * Ordinary feed backward pass of a neural network. Pass the error from output
   * layer to input layer and calculate the delta of kernel weights.
   *
   * @param error The backpropagated error.
   */
  void FeedBackward(const DataType& error)
  {
    BackwardConvolutionRule::conv(weights, error, delta);
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param gradient The calculated gradient.
   */
  void Gradient(DataType& gradient)
  {
    GradientConvolutionRule::Convolution(inputLayer.InputActivation(),
        outputLayer.Delta(), gradient);
  }

  //! Get the convolution kernel.
  DataType& Weights() const { return weights; }
  //! Modify the convolution kernel.
  DataType& Weights() { return weights; }

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

  //! Get the passed error in backward propagation.
  DataType& Delta() const { return delta; }
  //! Modify the passed error in backward propagation.
  DataType& Delta() { return delta; }

 private:
  //! Locally-stored kernel weights.
  DataType weights;

  //! Locally-stored inputlayer.
  InputLayerType& inputLayer;

  //! Locally-stored outputlayer.
  OutputLayerType& outputLayer;

  //! Locally-stored optimizer.
  OptimizerType* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;

  //! Locally-stored passed error in backward propagation.
  DataType delta;
};// class ConvConnection

}; // namespace ann
}; // namespace mlpack

#endif