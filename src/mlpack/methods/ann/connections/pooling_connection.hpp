/**
 * @file cnn_pooling_connection.hpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Implementation of the pooling connection between input layer and output layer
 * for the convolutional neural network.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_POOLING_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_POOLING_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/optimizer/steepest_descent.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/pooling_rules/max_pooling.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

namespace mlpack{
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the pooling connection class for the convolutional neural
 * network. The pooling connection connects input layer with the output layer
 * using the specified pooling rule.
 *
 * @tparam InputLayerType Type of the connected input layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam PoolingRule Type of the pooling strategy.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam DataType Type of data (arma::mat, arma::sp_mat or arma::cube).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename PoolingRule = MaxPooling,
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    typename DataType = arma::cube
>
class PoolingConnection
{
 public:
  /**
   * Create the PoolingConnection object using the specified input layer, output
   * layer, optimizer and pooling strategy.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param PoolingRule The strategy of pooling.
   */
  PoolingConnection(InputLayerType& inputLayer,
                    OutputLayerType& outputLayer,
                    PoolingRule pooling = PoolingRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(0),
      weights(0),
      pooling(pooling),
      delta(arma::zeros<DataType>(inputLayer.Delta().n_rows,
          inputLayer.Delta().n_cols, inputLayer.Delta().n_slices))
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, apply pooling to the
   * neurons (dense matrix) in the input layer.
   *
   * @param input Input data used for pooling.
   */
  template<typename eT>
  void FeedForward(const arma::Mat<eT>& input)
  {
    Pooling(input, outputLayer.InputActivation());
  }

  /**
   * Ordinary feed forward pass of a neural network, apply pooling to the
   * neurons (3rd order tensor) in the input layer.
   *
   * @param input Input data used for pooling.
   */
  template<typename eT>
  void FeedForward(const arma::Cube<eT>& input)
  {
    for (size_t s = 0; s < input.n_slices; s++)
      Pooling(input.slice(s), outputLayer.InputActivation().slice(s));
  }

  /**
   * Ordinary feed backward pass of a neural network. Apply unsampling to the
   * error in output layer (dense matrix) to pass the error to input layer.
   *
   * @param error The backpropagated error.
   */
  template<typename eT>
  void FeedBackward(const arma::Mat<eT>& error)
  {
    delta.zeros();
    Unpooling(inputLayer.InputActivation(), error, inputLayer.Delta());
  }

  /**
   * Ordinary feed backward pass of a neural network. Apply unsampling to the
   * error in output layer (3rd order tensor) to pass the error to input layer.
   *
   * @param error The backpropagated error.
   */
  template<typename eT>
  void FeedBackward(const arma::Cube<eT>& error)
  {
    delta.zeros();
    for (size_t s = 0; s < error.n_slices; s++)
    {
      Unpooling(inputLayer.InputActivation().slice(s), error.slice(s),
          delta.slice(s));
    }
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param gradient The calculated gradient.
   */
  template<typename GradientType>
  void Gradient(GradientType& /* unused */)
  {
    // Nothing to do here.
  }

  //! Get the weights.
  DataType& Weights() const { return *weights; }
  //! Modify the weights.
  DataType& Weights() { return *weights; }

  //! Get the input layer.
  InputLayerType& InputLayer() const { return inputLayer; }
  //! Modify the input layer.
  InputLayerType& InputLayer() { return inputLayer; }

  //! Get the output layer.
  OutputLayerType& OutputLayer() const { return outputLayer; }
  //! Modify the output layer.
  OutputLayerType& OutputLayer() { return outputLayer; }

  //! Get the optimizer.
  OptimizerType<PoolingConnection<InputLayerType,
                                  OutputLayerType,
                                  PoolingRule,
                                  OptimizerType,
                                  DataType>, DataType>& Optimzer() const
  {
    return *optimizer;
  }
  //! Modify the optimzer.
  OptimizerType<PoolingConnection<InputLayerType,
                                  OutputLayerType,
                                  PoolingRule,
                                  OptimizerType,
                                  DataType>, DataType>& Optimzer()
  {
    return *optimizer;
  }

  //! Get the passed error in backward propagation.
  DataType& Delta() const { return delta; }
  //! Modify the passed error in backward propagation.
  DataType& Delta() { return delta; }

  //! Get the pooling strategy.
  PoolingRule& Pooling() const { return pooling; }
  //! Modify the pooling strategy.
  PoolingRule& Pooling() { return pooling; }

 private:
  /**
   * Apply pooling to the input and store the results.
   *
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   */
  template<typename eT>
  void Pooling(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    const size_t rStep = input.n_rows / outputLayer.LayerRows();
    const size_t cStep = input.n_cols / outputLayer.LayerCols();

    for (size_t j = 0; j < input.n_cols; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows; i += rStep)
      {
        output(i / rStep, j / cStep) += pooling.Pooling(
            input(arma::span(i, i + rStep -1), arma::span(j, j + cStep - 1)));
      }
    }
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param input The input to be apply the unpooling rule.
   * @param output The pooled result.
   */
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& error,
                 arma::Mat<eT>& output)
  {
    const size_t rStep = input.n_rows / error.n_rows;
    const size_t cStep = input.n_cols / error.n_cols;

    arma::Mat<eT> unpooledError;
    for (size_t j = 0; j < input.n_cols; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows; i += rStep)
      {
        const arma::Mat<eT>& inputArea = input(arma::span(i, i + rStep - 1),
                                               arma::span(j, j + cStep - 1));

        pooling.Unpooling(inputArea, error(i / rStep, j / cStep),
            unpooledError);

        output(arma::span(i, i + rStep - 1),
            arma::span(j, j + cStep - 1)) += unpooledError;
      }
    }
  }

  //! Locally-stored input layer.
  InputLayerType& inputLayer;

  //! Locally-stored output layer.
  OutputLayerType& outputLayer;

  //! Locally-stored optimizer.
  OptimizerType<PoolingConnection<InputLayerType,
                                  OutputLayerType,
                                  PoolingRule,
                                  OptimizerType,
                                  DataType>, DataType>* optimizer;

  //! Locally-stored weight object.
  DataType* weights;

  //! Locally-stored pooling strategy.
  PoolingRule pooling;

  //! Locally-stored passed error in backward propagation.
  DataType delta;
}; // PoolingConnection class.

//! Connection traits for the pooling connection.
template<
    typename InputLayerType,
    typename OutputLayerType,
    typename PoolingRule,
    template<typename, typename> class OptimizerType,
    typename DataType
>
class ConnectionTraits<
    PoolingConnection<InputLayerType, OutputLayerType, PoolingRule,
    OptimizerType, DataType> >
{
 public:
  static const bool IsSelfConnection = false;
  static const bool IsFullselfConnection = false;
  static const bool IsPoolingConnection = true;
};

}; // namespace ann
}; // namespace mlpack

#endif