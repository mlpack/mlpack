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

//#include "visitor/delete_visitor.hpp"
//#include "visitor/delta_visitor.hpp"
//#include "visitor/output_parameter_visitor.hpp"
//#include "visitor/reset_visitor.hpp"

#include "init_rules/network_init.hpp"

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <ensmallen.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Defenition of a standard recurrent neural network container.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
    typename OutputLayerType = NegativeLogLikelihood<>,
    typename InitializationRuleType = RandomInitialization,
    typename InputType = arma::mat,
    typename OutputType = arma::mat>
class RNNType : public FFN<OutputLayerType,
                           InitializationRuleType,
                           InputType,
                           OutputType>
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = RNNType<OutputLayerType,
                              InitializationRuleType,
                              InputType,
                              OutputType>;

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
  RNNType(const size_t bpttTruncate = 0,
          OutputLayerType outputLayer = OutputLayerType(),
          InitializationRuleType initializeRule = InitializationRuleType());

  //! Copy constructor.
  RNNType(const RNNType&);

  //! Move constructor.
  RNNType(RNNType&&);

  //! Copy/move assignment operator.
  RNNType& operator=(RNNType);

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
  double Train(arma::Cube<typename InputType::elem_type> predictors,
               arma::Cube<typename InputType::elem_type> responses,
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
  double Train(InputType predictors,
               InputType responses,
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
  void Predict(InputType predictors,
               OutputType& results,
               const size_t batchSize = 128);

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

  /**
   * Evaluate the recurrent network with the given parameters. This function
   * is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   */
  double Evaluate(const OutputType& parameters);

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
  double Evaluate(const OutputType& parameters,
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
  double EvaluateWithGradient(const OutputType& parameters,
                              OutputType& gradient);

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
  double EvaluateWithGradient(const OutputType& parameters,
                              const size_t begin,
                              OutputType& gradient,
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
  void Gradient(const OutputType& parameters,
                const size_t begin,
                OutputType& gradient,
                const size_t batchSize);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

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
  template<typename PredictorsType, typename ResponsesType>
  void Forward(const PredictorsType& inputs, ResponsesType& results);

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
  template<typename PredictorsType, typename ResponsesType>
  void Forward(const PredictorsType& inputs ,
               ResponsesType& results,
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
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args)
  {
    network.template Add<LayerType>(args...);
    inputDimensionsAreSet = false;
  }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  //! TODO: if weights are set in this layer, we should copy them and update our
  //cached parameters
  void Add(Layer<InputType, OutputType>* layer)
  {
    network.Add(layer);
    inputDimensionsAreSet = false;
  }

  //! Get the network model.
  const std::vector<Layer<InputType, OutputType>*>& Network() const
  {
    return network.Network();
  }

  /**
   * Modify the network model.  Be careful!  If you change the structure of the
   * network or parameters for layers, its state may become invalid, so be sure
   * to call ResetParameters() afterwards.  Don't add any layers like this; use
   * `Add()` instead.
   */
  std::vector<Layer<InputType, OutputType>*>& Network()
  {
    return network.Network();
  }

  /**
   * Set the logical dimensions of the input.
   *
   * TODO: better comment.  You would call this when you want to, e.g., pass an
   * n-dimensional tensor, so that you can specify each of those n dimensions.
   */
  // Note: we don't need to invalidate any caches, because any forward pass will
  // already check if the input dimensions have changed.
  std::vector<size_t>& InputDimensions() { return inputDimensions; }
  //! Get the logical dimensions of the input.
  const std::vector<size_t>& InputDimensions() const
  {
    // The user may change the input dimensions, so we will have to propagate
    // these changes to the network.
    inputDimensionsAreSet = false;
    return inputDimensions;
  }

  /**
   * Set all the layers in the network to training mode, if `training` is
   * `true`, or set all the layers in the network to testing mode, if `training`
   * is `false`.
   */
  void SetNetworkMode(const bool training);

    //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return responses.n_slices; }

  //! Return the initial point for the optimization.
  const OutputType& Parameters() const { return parameters; }
  //! Modify the initial point for the optimization.
  OutputType& Parameters() { return parameters; }

  //! Get the matrix of responses to the input data points.
  const InputType& Responses() const { return responses; }
  //! Modify the matrix of responses to the input data points.
  InputType& Responses() { return responses; }

  //! Get the matrix of data points (predictors).
  const InputType& Predictors() const { return predictors; }
  //! Modify the matrix of data points (predictors).
  InputType& Predictors() { return predictors; }

 private:
  // Helper functions.

  /**
   * Prepare the network for the given data.
   * This function won't actually trigger training process.
   *
   * @param predictors Input data variables.
   * @param responses Outputs results from input data variables.
   */
  void ResetData(arma::Cube<typename InputType::elem_type> predictors,
                 arma::Cube<typename InputType::elem_type> responses);

  //! Use the InitializationPolicy to initialize all the weights in the network.
  void InitializeWeights();

  //! Make the memory of each layer point to the right place, by calling
  //! SetWeightPtr() on each layer.
  void SetLayerMemory();

  /**
   * Ensure that all the locally-cached information about the network is valid,
   * all parameter memory is initialized, and we can make forward and backward
   * passes.
   *
   * @param functionName Name of function to use if an exception is thrown.
   * @param inputDimensionality Given dimensionality of the input data.
   * @param setMode If true, the mode of the network will be set to the
   *     parameter given in `training`.  Otherwise the mode of the network is
   *     left unmodified.
   * @param training Mode to set the network to; `true` indicates the network
   *     should be set to training mode; `false` indicates testing mode.
   */
  void CheckNetwork(const std::string& functionName,
                    const size_t inputDimensionality,
                    const bool setMode = false,
                    const bool training = false);

  /**
   * Set the input and output dimensions of each layer in the network correctly.
   * The size of the input is taken, in case `inputDimensions` has not been set
   * otherwise (e.g. via `InputDimensions()`).
   */
  void UpdateDimensions(const std::string& functionName,
                        const size_t inputDimensionality);

  /**
   * Swap the content of this network with given network.
   *
   * @param network Desired source network.
   */
  void Swap(RNNType& network);

  //! Instantiated outputlayer used to evaluate the network.
  OutputLayerType outputLayer;

  //! Instantiated InitializationRule object for initializing the network
  //! parameter.
  InitializationRuleType initializeRule;

  //! All of the network is stored inside this multilayer.
  MultiLayer<InputType, OutputType> network;

  /**
   * Matrix of (trainable) parameters.  Each weight here corresponds to a layer,
   * and each layer's `parameters` member is an alias pointing to parameters in
   * this matrix.
   *
   * Note: although each layer may have its own InputType and OutputType,
   * ensmallen optimization requires everything to be stored in one matrix
   * object, so we have chosen OutputType.  This could be made more flexible
   * with a "wrapper" class implementing the Armadillo API.
   */
  OutputType parameters;

  //! Dimensions of input data.
  std::vector<size_t> inputDimensions;

  //! The matrix of data points (predictors).  This member is empty, except
  //! during training---we must store a local copy of the training data since
  //! the ensmallen optimizer will not provide training data.
  arma::Cube<typename InputType::elem_type> predictors;

  //! The matrix of responses to the input data points.  This member is empty,
  //! except during training.
  arma::Cube<typename InputType::elem_type> responses;

  //! The current error for the backward pass.
  OutputType networkOutput;
  OutputType networkDelta;
  OutputType error;

  //! The current evaluation mode (training or testing).
  bool training;

  //! If true, each layer has its memory properly set for a forward/backward
  //! pass.
  bool layerMemoryIsSet;

  //! If true, each layer has its inputDimensions properly set, and
  //! `totalInputSize` and `totalOutputSize` are valid.
  bool inputDimensionsAreSet;
}; // class RNNType

// Convenience typedefs.

/**
 * Standard Sigmoid-Layer using the logistic activation function.
 */
using RNN = RNNType<>;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "rnn_impl.hpp"

#endif
