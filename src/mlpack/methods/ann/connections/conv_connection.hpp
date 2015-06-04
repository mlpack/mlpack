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
#include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>

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
    typename BackwardConvolutionRule = FFTConvolution<FullConvolution>,
    typename GradientConvolutionRule = FFTConvolution<ValidConvolution>,
    typename DataType = arma::cube
>
class ConvConnection
{
 public:
  /**
   * Create the ConvConnection object using the specified input layer, output
   * layer, filter size, optimizer and weight initialization rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param filterSize the size of the filter.
   * @param OptimizerType The optimizer used to update the weight matrix.
   * @param WeightInitRule The weights initialization rule used to initialize
   * the weights matrix.
   */
  ConvConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 const size_t filterSize,
                 OptimizerType& optimizer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(&optimizer),
      ownsOptimizer(false)
  {
    weightInitRule.Initialize(weights, filterSize, filterSize,
        outputLayer.LayerSlices());
  }

  /**
   * Create the ConvConnection object using the specified input layer, output
   * layer, filter size and weight initialization rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param filterSize the size of the filter.
   * @param WeightInitRule The weights initialization rule used to initialize
   * the weights matrix.
   */
  ConvConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 const size_t filterSize,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(new OptimizerType()),
      ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, filterSize, filterSize,
        inputLayer.OutputMaps() * outputLayer.OutputMaps());
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
   *
   * @param input The input activation.
   */
  template<typename InputType>
  void FeedForward(const InputType& input)
  {
    for (size_t outputmap = 0; outputmap < outputLayer.OutputMaps(); outputmap++)
    {
      for (size_t inputmap = 0; inputmap < inputLayer.OutputMaps(); inputmap++)
      {
        InputType inputSlices = input.slices(
            inputmap * inputLayer.LayerSlices(),
            (inputmap * inputLayer.LayerSlices()) +
            inputLayer.LayerSlices() - 1);

        InputType output;
        ForwardConvolutionRule::Convolution(inputSlices,
            weights.slice(inputmap * outputLayer.OutputMaps() +
            outputmap), output);

        outputLayer.InputActivation().slices(
            (outputmap * inputLayer.LayerSlices()),
            (outputmap * inputLayer.LayerSlices()) +
            inputLayer.LayerSlices() - 1) += output;
      }
    }
  }

  /**
   * Ordinary feed backward pass of a neural network. Pass the error from output
   * layer to input layer and calculate the delta of kernel weights.
   *
   * @param error The backpropagated error.
   */
  template<typename eT>
  void FeedBackward(const arma::Cube<eT>& error)
  {
    delta = arma::zeros<arma::Cube<eT>>(inputLayer.InputActivation().n_rows,
                                        inputLayer.InputActivation().n_cols,
                                        inputLayer.InputActivation().n_slices);

    for (size_t outputmap = 0; outputmap < inputLayer.OutputMaps(); outputmap++)
    {
      for (size_t inputmap = 0; inputmap < outputLayer.OutputMaps(); inputmap++)
      {
        arma::Cube<eT> errorSlices = error.slices(inputmap *
            inputLayer.LayerSlices(), (inputmap * inputLayer.LayerSlices()) +
            inputLayer.LayerSlices() - 1);

        arma::Mat<eT> rotatedFilter;
        Rotate180(weights.slice(
            outputmap * outputLayer.OutputMaps() + inputmap), rotatedFilter);

        arma::Cube<eT> output;
        BackwardConvolutionRule::Convolution(errorSlices, rotatedFilter, output);

        delta.slices((outputmap * inputLayer.LayerSlices()),
            (outputmap * inputLayer.LayerSlices()) +
            inputLayer.LayerSlices() - 1) += output;
      }
    }
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Cube<eT>& gradient)
  {
    gradient = arma::zeros<arma::Cube<eT> >(weights.n_rows, weights.n_cols,
        weights.n_slices);

    for (size_t outputmap = 0, s = 0; outputmap < outputLayer.OutputMaps(); outputmap++)
    {
      for (size_t inputmap = 0; inputmap < inputLayer.OutputMaps(); inputmap++, s++)
      {
        arma::Cube<eT> inputSlices = inputLayer.InputActivation().slices(
            inputmap * inputLayer.LayerSlices(), (inputmap + 1) *
            inputLayer.LayerSlices() - 1);

        arma::Cube<eT> deltaSlices = outputLayer.Delta().slices(
            outputmap * inputLayer.LayerSlices(),
            (outputmap + 1) * inputLayer.LayerSlices() - 1);

        arma::Cube<eT> output;
        GradientConvolutionRule::Convolution(inputSlices, deltaSlices, output);

        for (size_t i = 0; i < output.n_slices; i++)
          gradient.slice(s) += output.slice(i);

        gradient.slice(s) /= inputLayer.LayerSlices();
      }
    }
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
  /*
   * Rotates a 3rd-order tesor counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  template<typename eT>
  void Rotate180(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = arma::Cube<eT>(input.n_rows, input.n_cols, input.n_slices);

    // * left-right flip, up-down flip */
    for (size_t s = 0; s < output.n_slices; s++)
      output.slice(s) = arma::fliplr(arma::flipud(input.slice(s)));
  }

  /*
   * Rotates a dense matrix counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  template<typename eT>
  void Rotate180(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    // * left-right flip, up-down flip */
    output = arma::fliplr(arma::flipud(input));
  }

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