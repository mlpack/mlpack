/**
 * @file methods/ann/rnn.hpp
 * @author Marcus Edel
 *
 * Definition of the RNN class, which implements recurrent neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RNN_HPP
#define MLPACK_METHODS_ANN_RNN_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

#include "ffn.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Definition of a standard recurrent neural network container.  A recurrent
 * neural network can handle recurrent layers (i.e. `RecurrentLayer`s), which
 * hold internal state and are passed sequences of data as inputs.
 *
 * As opposed to the standard `FFN`, which takes data in a matrix format where
 * each column is a data point, the `RNN` takes a cube format where each column
 * is a data point and each slice is a time step.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
    typename OutputLayerType = NegativeLogLikelihood,
    typename InitializationRuleType = RandomInitialization,
    typename MatType = arma::mat>
class RNN
{
 public:
  /**
   * Create the RNN object.
   *
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  RNN(const size_t bpttTruncate = 0,
      const bool single = false,
      OutputLayerType outputLayer = OutputLayerType(),
      InitializationRuleType initializeRule = InitializationRuleType());

  //! Copy constructor.
  RNN(const RNN&);

  //! Move constructor.
  RNN(RNN&&);

  //! Copy/move assignment operator.
  RNN& operator=(const RNN&);
  RNN& operator=(RNN&&);

  ~RNN();

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.template Add<LayerType>(args...); }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(Layer<MatType>* layer) { network.Add(layer); }

  //! Get the network model.
  const std::vector<Layer<MatType>*>& Network() const
  {
    return network.Network().Network();
  }

  /**
   * Modify the network model.  Be careful!  If you change the structure of the
   * network or parameters for layers, its state may become invalid, so be sure
   * to call ResetParameters() afterwards.  Don't add any layers like this; use
   * `Add()` instead.
   */
  std::vector<Layer<MatType>*>& Network()
  {
    return network.Network().Network();
  }

  /**
   * Train the recurrent network on the given input data using the given
   * optimizer.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackTypes>
  double Train(arma::Cube<typename MatType::elem_type> predictors,
               arma::Cube<typename MatType::elem_type> responses,
               OptimizerType& optimizer,
               CallbackTypes&&... callbacks);

  /**
   * Train the recurrent network on the given input data. By default, the
   * RMSProp optimization algorithm is used, but others can be specified
   * (such as ens::SGD).
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param predictors Input training variables.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param responses Outputs results from input training variables.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType = ens::RMSProp, typename... CallbackTypes>
  double Train(arma::Cube<typename MatType::elem_type> predictors,
               arma::Cube<typename MatType::elem_type> responses,
               CallbackTypes&&... callbacks);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * output layer function.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   * @param batchSize Batch size to use for prediction.
   */
  void Predict(arma::Cube<typename MatType::elem_type> predictors,
               arma::Cube<typename MatType::elem_type>& results,
               const size_t batchSize = 128);

  // Return the nujmber of weights in the model.
  size_t WeightSize() { return network.WeightSize(); }

  /**
   * Set the logical dimensions of the input.
   *
   * TODO: better comment.  You would call this when you want to, e.g., pass an
   * n-dimensional tensor, so that you can specify each of those n dimensions.
   */
  // Note: we don't need to invalidate any caches, because any forward pass will
  // already check if the input dimensions have changed.
  std::vector<size_t>& InputDimensions() { return network.InputDimensions(); }
  //! Get the logical dimensions of the input.
  const std::vector<size_t>& InputDimensions() const
  {
    return network.InputDimensions();
  }

  //! Return the initial point for the optimization.
  const MatType& Parameters() const { return network.Parameters(); }
  //! Modify the initial point for the optimization.
  MatType& Parameters() { return network.Parameters(); }

  //! Return the number of steps allowed for BPTT.
  size_t Rho() const { return rho; }
  //! Modify the number of steps allowed for BPTT.
  size_t& Rho() { return rho; }

  void Reset(const size_t inputDimensionality = 0);

  /**
   * Set all the layers in the network to training mode, if `training` is
   * `true`, or set all the layers in the network to testing mode, if `training`
   * is `false`.
   */
  void SetNetworkMode(const bool training) { network.SetNetworkMode(training); }

  /**
   * Perform the forward pass of the data in real batch mode.
   *
   * Forward and Backward should be used as a pair, and they are designed mainly
   * for advanced users. User should try to use Predict and Train unless those
   * two functions can't satisfy some special requirements.
   *
   * @param inputs The input data.
   * @param results The predicted results.
   */
  void Forward(const arma::Cube<typename MatType::elem_type>& inputs,
               arma::Cube<typename MatType::elem_type>& results);

  /**
   * Perform a partial forward pass of the data.
   *
   * This function is meant for the cases when users require a forward pass only
   * through certain layers and not the entire network.
   *
   * @param inputs The input data for the specified first layer.
   * @param results The predicted results from the specified last layer.
   * @param begin The index of the first layer.
   * @param end The index of the last layer.
   */
  void Forward(const arma::Cube<typename MatType::elem_type>& inputs,
               arma::Cube<typename MatType::elem_type>& results,
               const size_t begin,
               const size_t end);

  /**
   * Perform the backward pass of the data in real batch mode.
   *
   * Forward and Backward should be used as a pair, and they are designed mainly
   * for advanced users. User should try to use Predict and Train unless those
   * two functions can't satisfy some special requirements.
   *
   * @param inputs Inputs of current pass.
   * @param targets The training target.
   * @param gradients Computed gradients.
   * @return Training error of the current pass.
   */
  template<typename PredictorsType,
           typename TargetsType,
           typename GradientsType>
  double Backward(const PredictorsType& inputs,
                  const TargetsType& targets,
                  GradientsType& gradients);

  /**
   * Evaluate the recurrent network with the given predictors and responses.
   * This functions is usually used to monitor progress while training.
   *
   * @param predictors Input variables.
   * @param responses Target outputs for input variables.
   */
  template<typename PredictorsType, typename ResponsesType>
  double Evaluate(const PredictorsType& predictors,
                  const ResponsesType& responses);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //
  // Only ensmallen utility functions for training are found below here.
  // They generally aren't useful otherwise.
  //

  /**
   * Evaluate the recurrent network with the given parameters. This function
   * is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   */
  double Evaluate(const MatType& parameters);

   /**
   * Evaluate the recurrent network with the given parameters, but using only
   * a number of data points. This is useful for optimizers such as SGD, which
   * require a separable objective function.
   *
   * Note that the network may return different results depending on the mode it
   * is in (see `SetNetworkMode()`).
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */
  double Evaluate(const MatType& parameters,
                  const size_t begin,
                  const size_t batchSize);

  /**
   * Evaluate the recurrent network with the given parameters.
   * This function is usually called by the optimizer to train the model.
   * This just calls the overload of EvaluateWithGradient() with batchSize = 1.
   *
   * @param parameters Matrix model parameters.
   * @param gradient Matrix to output gradient into.
   */
  template<typename GradType>
  double EvaluateWithGradient(const MatType& parameters,
                              GradType& gradient);

   /**
   * Evaluate the recurrent network with the given parameters, but using only
   * a number of data points. This is useful for optimizers such as SGD, which
   * require a separable objective function.
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param gradient Matrix to output gradient into.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */
  template<typename GradType>
  double EvaluateWithGradient(const MatType& parameters,
                              const size_t begin,
                              GradType& gradient,
                              const size_t batchSize);

  /**
   * Evaluate the gradient of the recurrent network with the given parameters,
   * and with respect to only a number of points in the dataset. This is useful
   * for optimizers such as SGD, which require a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param begin Index of the starting point to use for objective function
   *        gradient evaluation.
   * @param gradient Matrix to output gradient into.
   * @param batchSize Number of points to be processed as a batch for objective
   *        function gradient evaluation.
   */
  template<typename GradType>
  void Gradient(const MatType& parameters,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize);

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return predictors.n_cols; }

  void Shuffle();

  /**
   * Prepare the network for the given data.
   * This function won't actually trigger training process.
   *
   * @param predictors Input data variables.
   * @param responses Outputs results from input data variables.
   */
  void ResetData(arma::Cube<typename MatType::elem_type> predictors,
                 arma::Cube<typename MatType::elem_type> responses);

 private:
  // Helper functions.

  void ResetMemoryState(const size_t memorySize, const size_t batchSize);
  void SetPreviousStep(const size_t step);
  void SetCurrentStep(const size_t step);

  size_t rho;
  bool single;

  //! The network itself is stored in this FFN object.  Note that this network
  //! may contain recursive layers, and thus we will be responsible for
  //! occasionally resetting any memory cells.
  FFN<OutputLayerType, InitializationRuleType, MatType> network;

  //! The matrix of data points (predictors).  This member is empty, except
  //! during training---we must store a local copy of the training data since
  //! the ensmallen optimizer will not provide training data.
  arma::Cube<typename MatType::elem_type> predictors;

  //! The matrix of responses to the input data points.  This member is empty,
  //! except during training.
  arma::Cube<typename MatType::elem_type> responses;
}; // class RNNType

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "rnn_impl.hpp"

#endif
