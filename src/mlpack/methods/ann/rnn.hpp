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

#include <mlpack/core.hpp>

#include "ffn.hpp"

namespace mlpack {

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
   * @param bpttSteps Number of time steps to use for BPTT (backpropagation
   *      through time) when training.
   * @param single If true, then the network will expect only a single timestep
   *      for responses.  (That is, every input sequence only has one single
   *      output; so, `responses.n_slices` should be 1 when calling `Train()`.)
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  RNN(const size_t bpttSteps = 0,
      const bool single = false,
      OutputLayerType outputLayer = OutputLayerType(),
      InitializationRuleType initializeRule = InitializationRuleType());

  //! Copy constructor.
  RNN(const RNN&);
  //! Move constructor.
  RNN(RNN&&);
  //! Copy operator.
  RNN& operator=(const RNN&);
  //! Move assignment operator.
  RNN& operator=(RNN&&);

  //! Destroy the RNN and release any memory it is holding.
  ~RNN();

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <typename LayerType, typename... Args>
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
    return network.Network();
  }

  /**
   * Train the recurrent network on the given input data using the given
   * optimizer.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
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
  typename MatType::elem_type Train(
      arma::Cube<typename MatType::elem_type> predictors,
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
  typename MatType::elem_type Train(
      arma::Cube<typename MatType::elem_type> predictors,
      arma::Cube<typename MatType::elem_type> responses,
      CallbackTypes&&... callbacks);

  /**
   * Train the recurrent network on the given input data using the given
   * optimizer, given that input sequences may have different lengths.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * Note that due to shuffling, training will make a copy of the data, unless
   * you use `std::move()` to pass the `predictors` and `responses` (that is,
   * `Train(std::move(predictors), std::move(responses))`).
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param sequenceLengths Length of each input sequences.  Should have size
   *     `predictors.n_cols`, and all values should be less than or equal to
   *     `predictors.n_slices`.
   * @param optimizer Instantiated optimizer used to train the model.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackTypes>
  typename MatType::elem_type Train(
      arma::Cube<typename MatType::elem_type> predictors,
      arma::Cube<typename MatType::elem_type> responses,
      arma::urowvec sequenceLengths,
      OptimizerType& optimizer,
      CallbackTypes&&... callbacks);

  /**
   * Train the recurrent network on the given input data, given that each input
   * sequence may have a different length.  By default, the RMSProp optimization
   * algorithm is used, but others can be specified (such as ens::SGD).
   *
   * When passing sequences with different lengths, the batch size of the
   * optimizer must be set to 1; if it is not, an exception will be thrown
   * during training.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * Note that due to shuffling, training will make a copy of the data, unless
   * you use `std::move()` to pass the `predictors` and `responses` (that is,
   * `Train(std::move(predictors), std::move(responses))`).
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param sequenceLengths Length of each input sequences.  Should have size
   *     `predictors.n_cols`, and all values should be less than or equal to
   *     `predictors.n_slices`.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType = ens::RMSProp, typename... CallbackTypes>
  typename MatType::elem_type Train(
      arma::Cube<typename MatType::elem_type> predictors,
      arma::Cube<typename MatType::elem_type> responses,
      arma::urowvec sequenceLengths,
      CallbackTypes&&... callbacks);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * output layer function.
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   * @param batchSize Batch size to use for prediction.
   */
  void Predict(const arma::Cube<typename MatType::elem_type>& predictors,
               arma::Cube<typename MatType::elem_type>& results,
               const size_t batchSize = 128);

  /**
   * Predict the responses to a given set of predictors, given that each
   * sequence can have a different length. The responses will reflect the output
   * of the given output layer as returned by the output layer function.
   *
   * Slices of column `i` of `results` at time indexes greater than
   * `sequenceLengths[i]` should not be considered valid predictions.
   *
   * The batch size is limited to 1 when predicting on sequences of different
   * lengths.
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   */
  void Predict(const arma::Cube<typename MatType::elem_type>& predictors,
               arma::Cube<typename MatType::elem_type>& results,
               const arma::urowvec& sequenceLengths);

  // Return the nujmber of weights in the model.
  size_t WeightSize() { return network.WeightSize(); }

  /**
   * Set the logical dimensions of the input.  `Train()` and `Predict()` expect
   * data to be passed such that one point corresponds to one column, but this
   * data is allowed to be an arbitrary higher-order tensor.
   *
   * So, if the input is meant to be 28x28x3 images, then the input data to
   * `Train()` or `Predict()` should have 28*28*3 = 2352 rows, and
   * `InputDImensions()` should be set to `{ 28, 28, 3}`.  Then, the layers of
   * the network will interpret each input point as a 3-dimensional image
   * instead of a 1-dimensional vector.
   *
   * If `InputDimensions()` is left unset before training, the data will be
   * assumed to be a 1-dimensional vector.
   */
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
  size_t BPTTSteps() const { return bpttSteps; }
  //! Modify the number of steps allowed for BPTT.
  size_t& BPTTSteps() { return bpttSteps; }

  /**
   * Reset the stored data of the network entirely.  This reset all weights of
   * each layer using `InitializationRuleType`, and prepares the network to
   * accept a (flat 1-d) input size of `inputDimensionality` (if passed), or
   * whatever input size has been set with `InputDimensions()`.
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
  void SetNetworkMode(const bool training) { network.SetNetworkMode(training); }

  /**
   * Evaluate the recurrent network with the given predictors and responses.
   * This functions is usually used to monitor progress while training.
   *
   * @param predictors Input variables.
   * @param responses Target outputs for input variables.
   */
  typename MatType::elem_type Evaluate(
      const arma::Cube<typename MatType::elem_type>& predictors,
      const arma::Cube<typename MatType::elem_type>& responses);

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
  typename MatType::elem_type Evaluate(const MatType& parameters);

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
  typename MatType::elem_type Evaluate(const MatType& parameters,
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
  typename MatType::elem_type EvaluateWithGradient(const MatType& parameters,
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
  typename MatType::elem_type EvaluateWithGradient(const MatType& parameters,
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

  /**
   * Note: this function is implement so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Shuffle the order of function visitation.  (This is equivalent to shuffling
   * the dataset during training.)
   */
  void Shuffle();

  /**
   * Prepare the network for the given data.
   * This function won't actually trigger training process.
   *
   * @param predictors Input data variables.
   * @param responses Outputs results from input data variables.
   * @param sequenceLengths (Optional) sequence length for each predictor
   *     sequence.
   */
  void ResetData(arma::Cube<typename MatType::elem_type> predictors,
                 arma::Cube<typename MatType::elem_type> responses,
                 arma::urowvec sequenceLengths = arma::urowvec());

 private:
  // Helper functions.

  /**
   * Iterate over all layers and reset the recurrent layers' states.  Prepare
   * each recurrent layer to store up to `memorySize` previous states, operating
   * with a batch size of `batchSize`.
   */
  void ResetMemoryState(const size_t memorySize, const size_t batchSize);

  //! Set the current step index of all recurrent layers to `step`.
  void SetCurrentStep(const size_t step, const bool end);

  //! Number of timesteps to consider for backpropagation through time (BPTT).
  size_t bpttSteps;
  //! Whether the network expects only one single response per sequence, or one
  //! response per time step.
  bool single;

  //! The network itself is stored in this FFN object.  Note that this network
  //! may contain recursive layers, and thus we will be responsible for
  //! occasionally resetting any memory cells.
  FFN<OutputLayerType, InitializationRuleType, MatType> network;

  // The matrix of data points (predictors).  These members are empty, except
  // during training---we must store a local copy of the training data since
  // the ensmallen optimizer will not provide training data.
  arma::Cube<typename MatType::elem_type> predictors;

  // The matrix of responses to the input data points.  This member is empty,
  // except during training.
  arma::Cube<typename MatType::elem_type> responses;

  // The length of each input sequence.  If this is empty, then every sequence
  // is assuemd to have the same length (`predictors.n_slices`).
  arma::urowvec sequenceLengths;
}; // class RNNType

} // namespace mlpack

// Include implementation.
#include "rnn_impl.hpp"

#endif
