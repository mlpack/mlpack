/**
 * @file methods/ann/ffn.hpp
 * @author Marcus Edel
 * @author Shangtong Zhang
 *
 * Definition of the FFN class, which implements feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FFN_HPP
#define MLPACK_METHODS_ANN_FFN_HPP

#include <mlpack/core.hpp>

#include "forward_decls.hpp"
#include "init_rules/init_rules.hpp"
#include "loss_functions/loss_functions.hpp"

#include <ensmallen.hpp>

namespace mlpack {

/**
 * Implementation of a standard feed forward network.  Any layer that inherits
 * from the base `Layer` class can be added to this model.  For recursive neural
 * networks, see the `RNN` class.
 *
 * In general, a network can be created by using the `Add()` method to add
 * layers to the network.  Then, training can be performed with `Train()`, and
 * data points can be passed through the trained network with `Predict()`.
 *
 * Although the actual types passed as input will be matrix objects with one
 * data point per column, each data point can be a tensor of arbitrary shape.
 * If data points are not 1-dimensional vectors, then set the shape of the input
 * with `InputDimensions()` before calling `Train()`.
 *
 * More granular functionality is available with `Forward()`, Backward()`, and
 * `Evaluate()`, or even by accessing the individual layers directly with
 * `Network()`.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam MatType Type of matrix to be given as input to the network.
 * @tparam MatType Type of matrix to be produced as output from the last
 *     layer.
 */
template<
    typename OutputLayerType = NegativeLogLikelihood,
    typename InitializationRuleType = RandomInitialization,
    typename MatType = arma::mat>
class FFN
{
 public:
  /**
   * Create the FFN object.
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
  FFN(OutputLayerType outputLayer = OutputLayerType(),
      InitializationRuleType initializeRule = InitializationRuleType());

  //! Copy constructor.
  FFN(const FFN& other);
  //! Move constructor.
  FFN(FFN&& other);
  //! Copy operator.
  FFN& operator=(const FFN& other);
  //! Move assignment operator.
  FFN& operator=(FFN&& other);

  /**
   * Add a new layer to the model.
   *
   * @param args The layer parameter.
   */
  template <typename LayerType, typename... Args>
  void Add(Args... args)
  {
    network.template Add<LayerType>(args...);
    inputDimensionsAreSet = false;
  }

  /**
   * Add a new layer to the model.  Note that any trainable weights of this
   * layer will be reset!  (Any constant parameters are kept.)
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(Layer<MatType>* layer)
  {
    network.Add(layer);
    inputDimensionsAreSet = false;
  }

  //! Get the layers of the network.
  const std::vector<Layer<MatType>*>& Network() const
  {
    return network.Network();
  }

  /**
   * Modify the network model.  Be careful!  If you change the structure of the
   * network or parameters for layers, its state may become invalid, and the
   * next time it is used for any operation the parameters will be reset.
   *
   * Don't add any layers like this; use `Add()` instead.
   */
  std::vector<Layer<MatType>*>& Network()
  {
    // We can no longer make any assumptions... the user may change anything.
    inputDimensionsAreSet = false;
    layerMemoryIsSet = false;

    return network.Network();
  }

  /**
   * Train the feedforward network on the given input data using the given
   * optimizer.
   *
   * If no parameters have ever been set (e.g. if `Parameters()` is an empty
   * matrix), or if the parameters' size does not match the number of weights
   * needed for the current input size (as given by `predictors` and optionally
   * set further by `InputDimensions()`), then the network will be initialized
   * using `InitializeRuleType`.
   *
   * If parameters are the right size for the given `predictors` and
   * `InputDimensions()`, then the existing parameters will be used as a
   * starting point.  (If you want to reinitialize, first call `Reset()`.)
   *
   * Note that due to shuffling, training will make a copy of the data, unless
   * you use `std::move()` to pass the `predictors` and `responses` (that is,
   * `Train(std::move(predictors), std::move(responses))`).
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
  typename MatType::elem_type Train(MatType predictors,
                                    MatType responses,
                                    OptimizerType& optimizer,
                                    CallbackTypes&&... callbacks);

  /**
   * Train the feedforward network on the given input data. By default, the
   * RMSProp optimization algorithm is used, but others can be specified
   * (such as ens::SGD).
   *
   * If no parameters have ever been set (e.g. if `Parameters()` is an empty
   * matrix), or if the parameters' size does not match the number of weights
   * needed for the current input size (as given by `predictors` and optionally
   * set further by `InputDimensions()`), then the network will be initialized
   * using `InitializeRuleType`.
   *
   * If parameters are the right size for the given `predictors` and
   * `InputDimensions()`, then the existing parameters will be used as a
   * starting point.  (If you want to reinitialize, first call `Reset()`.)
   *
   * Note that due to shuffling, training will make a copy of the data, unless
   * you use `std::move()` to pass the `predictors` and `responses` (that is,
   * `Train(std::move(predictors), std::move(responses))`).
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
  typename MatType::elem_type Train(MatType predictors,
                                    MatType responses,
                                    CallbackTypes&&... callbacks);

  /**
   * Predict the responses to a given set of predictors. The responses will be
   * the output of the output layer when `predictors` is passed through the
   * whole network (`OutputLayerType`).
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   * @param batchSize Batch size to use for prediction.
   */
  void Predict(const MatType& predictors,
               MatType& results,
               const size_t batchSize = 128);

  // Return the number of weights in the model.
  size_t WeightSize();

  /**
   * Set the logical dimensions of the input.  `Train()` and `Predict()` expect
   * data to be passed such that one point corresponds to one column, but this
   * data is allowed to be an arbitrary higher-order tensor.
   *
   * So, if the input is meant to be 28x28x3 images, then the
   * input data to `Train()` or `Predict()` should have 28*28*3 = 2352 rows, and
   * `InputDimensions()` should be set to `{ 28, 28, 3 }`.  Then, the layers of
   * the network will interpret each input point as a 3-dimensional image
   * instead of a 1-dimensional vector.
   *
   * If `InputDimensions()` is left unset before training, the data will be
   * assumed to be a 1-dimensional vector.
   */
  std::vector<size_t>& InputDimensions()
  {
    // The user may change the input dimensions, so we will have to propagate
    // these changes to the network.
    inputDimensionsAreSet = false;
    return inputDimensions;
  }
  //! Get the logical dimensions of the input.
  const std::vector<size_t>& InputDimensions() const { return inputDimensions; }

  //! Return the current set of weights.  These are linearized: this contains
  //! the weights of every layer.
  const MatType& Parameters() const { return parameters; }
  //! Modify the current set of weights.  These are linearized: this contains
  //! the weights of every layer.  Be careful!  If you change the shape of
  //! `parameters` to something incorrect, it may be re-initialized the next
  //! time a forward pass is done.
  MatType& Parameters() { return parameters; }

  /**
   * Reset the stored data of the network entirely.  This resets all weights of
   * each layer using `InitializationRuleType`, and prepares the network to
   * accept a (flat 1-d) input size of `inputDimensionality` (if passed), or
   * whatever input size has been set with `InputDimensions()`.
   *
   * If no input size has been set with `InputDimensions()`, and
   * `inputDimensionality` is 0, an exception will be thrown, since an empty
   * input size is invalid.
   *
   * This also resets the mode of the network to prediction mode (not training
   * mode).  See `SetNetworkMode()` for more information.
   */
  void Reset(const size_t inputDimensionality = 0);

  /**
   * Set all the layers in the network to training mode, if `training` is
   * `true`, or set all the layers in the network to testing mode, if `training`
   * is `false`.
   */
  void SetNetworkMode(const bool training);

  /**
   * Perform a manual forward pass of the data.
   *
   * `Forward()` and `Backward()` should be used as a pair, and they are
   * designed mainly for advanced users. You should try to use `Predict()` and
   * `Train()`, if you can.
   *
   * @param inputs The input data.
   * @param results The predicted results.
   */
  void Forward(const MatType& inputs, MatType& results);

  /**
   * Perform a manual partial forward pass of the data.
   *
   * This function is meant for the cases when users require a forward pass only
   * through certain layers and not the entire network.  `Forward()` and
   * `Backward()` should be used as a pair, and they are designed mainly for
   * advanced users. You should try to use `Predict()` and `Train()`, if you
   * can.
   *
   * @param inputs The input data for the specified first layer.
   * @param results The predicted results from the specified last layer.
   * @param begin The index of the first layer.
   * @param end The index of the last layer.
   */
  void Forward(const MatType& inputs,
               MatType& results,
               const size_t begin,
               const size_t end);

  /**
   * Perform a manual backward pass of the data.
   *
   * `Forward()` and `Backward()` should be used as a pair, and they are
   * designed mainly for advanced users. You should try to use `Predict()` and
   * `Train()` instead, if you can.
   *
   * @param inputs Inputs of current pass.
   * @param targets The training target.
   * @param gradients Computed gradients.
   * @return Training error of the current pass.
   */
  typename MatType::elem_type Backward(const MatType& inputs,
                                       const MatType& targets,
                                       MatType& gradients);

  /**
   * Evaluate the feedforward network with the given predictors and responses.
   * This functions is usually used to monitor progress while training.
   *
   * @param predictors Input variables.
   * @param responses Target outputs for input variables.
   */
  typename MatType::elem_type Evaluate(const MatType& predictors,
                                       const MatType& responses);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //
  // Only ensmallen utility functions for training are found below here.
  // They aren't generally useful otherwise.
  //

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the feedforward network with the given parameters.
   *
   * @param parameters Matrix model parameters.
   */
  typename MatType::elem_type Evaluate(const MatType& parameters);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the feedforward network with the given parameters, but using only
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
  typename MatType::elem_type Evaluate(const MatType& parameters,
                                       const size_t begin,
                                       const size_t batchSize);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the feedforward network with the given parameters.
   * This function is usually called by the optimizer to train the model.
   * This just calls the overload of EvaluateWithGradient() with batchSize = 1.
   *
   * @param parameters Matrix model parameters.
   * @param gradient Matrix to output gradient into.
   */
  typename MatType::elem_type EvaluateWithGradient(const MatType& parameters,
                                                   MatType& gradient);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the feedforward network with the given parameters, but using only
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
  typename MatType::elem_type EvaluateWithGradient(const MatType& parameters,
                                                   const size_t begin,
                                                   MatType& gradient,
                                                   const size_t batchSize);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the gradient of the feedforward network with the given parameters,
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
  void Gradient(const MatType& parameters,
                const size_t begin,
                MatType& gradient,
                const size_t batchSize);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Return the number of separable functions (the number of predictor points).
   */
  size_t NumFunctions() const { return responses.n_cols; }

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Shuffle the order of function visitation.  (This is equivalent to shuffling
   * the dataset during training.)
   */
  void Shuffle();

  /**
   * Prepare the network for training on the given data.
   *
   * This function won't actually trigger the training process, and is
   * generally only useful internally.
   *
   * @param predictors Input data variables.
   * @param responses Outputs results from input data variables.
   */
  void ResetData(MatType predictors, MatType responses);

 private:
  // Helper functions.

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
   * otherwise (e.g. via `InputDimensions()`).  If `InputDimensions()` is not
   * empty, then `inputDimensionality` is ignored.
   */
  void UpdateDimensions(const std::string& functionName,
                        const size_t inputDimensionality = 0);

  /**
   * Check if the optimizer has MaxIterations() parameter, if it does then check
   * if its value is less than the number of datapoints in the dataset.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param optimizer optimizer used in the training process.
   * @param samples Number of datapoints in the dataset.
   */
  template<typename OptimizerType>
  std::enable_if_t<
      ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
  WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;

  /**
   * Check if the optimizer has MaxIterations() parameter; if it doesn't then
   * simply return from the function.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param optimizer optimizer used in the training process.
   * @param samples Number of datapoints in the dataset.
   */
  template<typename OptimizerType>
  std::enable_if_t<
      !ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
  WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;

  //! Instantiated output layer used to evaluate the network.
  OutputLayerType outputLayer;

  //! Instantiated InitializationRule object for initializing the network
  //! parameter.
  InitializationRuleType initializeRule;

  //! All of the network is stored inside this multilayer.
  MultiLayer<MatType> network;

  /**
   * Matrix of (trainable) parameters.  Each weight here corresponds to a layer,
   * and each layer's `parameters` member is an alias pointing to parameters in
   * this matrix.
   *
   * Note: although each layer may have its own MatType and MatType,
   * ensmallen optimization requires everything to be stored in one matrix
   * object, so we have chosen MatType.  This could be made more flexible
   * with a "wrapper" class implementing the Armadillo API.
   */
  MatType parameters;

  //! Dimensions of input data.
  std::vector<size_t> inputDimensions;

  //! The matrix of data points (predictors).  This member is empty, except
  //! during training---we must store a local copy of the training data since
  //! the ensmallen optimizer will not provide training data.
  MatType predictors;

  //! The matrix of responses to the input data points.  This member is empty,
  //! except during training.
  MatType responses;

  //! Locally-stored output of the network from a forward pass; used by the
  //! backward pass.
  MatType networkOutput;
  //! Locally-stored output of the backward pass; used by the gradient pass.
  MatType networkDelta;
  //! Locally-stored error of the backward pass; used by the gradient pass.
  MatType error;

  //! If true, each layer has its memory properly set for a forward/backward
  //! pass.
  bool layerMemoryIsSet;

  //! If true, each layer has its inputDimensions properly set, and
  //! `totalInputSize` and `totalOutputSize` are valid.
  bool inputDimensionsAreSet;

  // RNN will call `CheckNetwork()`, which is private.
  friend class RNN<OutputLayerType, InitializationRuleType, MatType>;
}; // class FFN

} // namespace mlpack

// Include implementation.
#include "ffn_impl.hpp"

#endif
