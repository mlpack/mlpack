/**
 * @file cnn_bias_connection.hpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Implementation of the connection between bias layer and other layer.
 */
#ifndef __MLPACK_METHODS_ANN_CONNECTIONS_BIAS_CONNECTION_HPP
#define __MLPACK_METHODS_ANN_CONNECTIONS_BIAS_CONNECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the bias connection class. The bias connection connects
 * bias layer and other layer.
 *
 * @tparam InputLayerType Type of the connected input layer. It must be a bias
 * layer.
 * @tparam OutputLayerType Type of the connected output layer.
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weights matrix.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 */
template<
    typename InputLayerType,
    typename OutputLayerType,
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = NguyenWidrowInitialization,
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
                 OptimizerType<BiasConnection<InputLayerType,
                                               OutputLayerType,
                                               OptimizerType,
                                               WeightInitRule,
                                               MatType>, MatType>& optimizer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inputLayer(inputLayer),
      outputLayer(outputLayer),
      optimizer(&optimizer),
      ownsOptimizer(false)
  {
    weightInitRule.Initialize(weights, outputLayer.OutputMaps(), 1);
  }

  /**
   * Create the BiasConnection object using the specified input layer, output
   * layer and weight initialize rule.
   *
   * @param InputLayerType The input layer which is connected with the output
   * layer.
   * @param OutputLayerType The output layer which is connected with the input
   * layer.
   * @param WeightInitRule The weight initialize rule used to initialize the
   * weight matrix.
   */
  BiasConnection(InputLayerType& inputLayer,
                 OutputLayerType& outputLayer,
                 WeightInitRule weightInitRule = WeightInitRule()) :
    inputLayer(inputLayer),
    outputLayer(outputLayer),
    optimizer(new OptimizerType<BiasConnection<InputLayerType,
                                               OutputLayerType,
                                               OptimizerType,
                                               WeightInitRule,
                                               MatType>, MatType>(*this)),
    ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outputLayer.OutputMaps(), 1);
  }

  /**
   * Delete the bias connection object and its optimizer.
   */
  ~BiasConnection()
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
    Forward(outputLayer.InputActivation(), input);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param error The backpropagated error.
   */
  template<typename eT>
  void FeedBackward(const arma::Cube<eT>& error)
  {
    delta = MatType(outputLayer.OutputMaps(), 1);
    for (size_t s = 0; s < error.n_slices; s++)
    {
      delta(s, 0) = weights(s, 0) * arma::accu(error.slice(s));
    }
  }

  template<typename eT>
  void FeedBackward(const arma::Mat<eT>& error)
  {
    delta = weights.t() * error;
  }

  /**
   * Calculate the gradient (dense matrix) using the output delta and the input
   * activation.
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>& gradient)
  {
    arma::Cube<eT> grad;
    Gradient(grad);
    gradient = grad.slice(0);
  }

  /*
   * Calculate the gradient (3rd order tensor) using the output delta
   * (3rd order tensor) and the input activation (3rd order tensor).
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
  OptimizerType<BiasConnection<InputLayerType,
                               OutputLayerType,
                               OptimizerType,
                               WeightInitRule,
                               MatType>, MatType>& Optimzer() const
  {
    return *optimizer;
  }

  //! Modify the optimzer.
  OptimizerType<BiasConnection<InputLayerType,
                               OutputLayerType,
                               OptimizerType,
                               WeightInitRule,
                               MatType>, MatType>& Optimzer()
  {
    return *optimizer;
  }

  //! Get the detla.
  MatType& Delta() const { return delta; }
  //! Modify the delta.
  MatType& Delta() { return delta; }

 private:
  /*
   * Calculate the gradient using the output delta (3rd order tensor) and the
   * input activation (3rd order tensor).
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(arma::Cube<eT>& /* unused */, arma::Cube<eT>& gradient)
  {
    gradient = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    for (size_t s = 0; s < outputLayer.OutputMaps(); s++)
    {
      gradient.slice(0)(s, 0) = arma::accu(outputLayer.Delta().slice(s)) *
          inputLayer.InputActivation()(s, 0);
    }
  }

  /*
   * Calculate the gradient (3rd order tensor) using the output delta
   * (dense matrix) and the input activation (dense matrix).
   *
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(arma::Mat<eT>& /* unused */, arma::Cube<eT>& gradient)
  {
    gradient = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    Gradient(gradient.slice(0));
  }

    /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f using a dense matrix as
   * input.
   *
   * @param input Input data used for evaluating the specified activity function.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& /* unused */, const arma::Mat<eT>& input)
  {
    for (size_t s = 0; s < outputLayer.OutputMaps(); s++)
      outputLayer.InputActivation().slice(s) += (weights(s, 0) * input(s, 0));
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f using a dense matrix as
   * input.
   *
   * @param input Input data used for evaluating the specified activity function.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& /* unused */, const arma::Mat<eT>& input)
  {
    outputLayer.InputActivation() += weights % input;
  }

  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored connected input layer object.
  InputLayerType& inputLayer;

  //! Locally-stored connected output layer object.
  OutputLayerType& outputLayer;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<BiasConnection<InputLayerType,
                               OutputLayerType,
                               OptimizerType,
                               WeightInitRule,
                               MatType>, MatType>* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;

  //! Locally-stored detla object that holds the calculated delta.
  MatType delta;
}; // class BiasConnection

}; // namespace ann
}; // namespace mlpack

#endif