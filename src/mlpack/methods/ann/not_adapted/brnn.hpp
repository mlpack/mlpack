/**
 * @file methods/ann/brnn.hpp
 * @author Saksham Bansal
 *
 * Definition of the BRNN class, which implements bidirectional recurrent
 * neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_BRNN_HPP
#define MLPACK_METHODS_ANN_BRNN_HPP

#include <mlpack/prereqs.hpp>

#include "visitor/delete_visitor.hpp"
#include "visitor/delta_visitor.hpp"
#include "visitor/copy_visitor.hpp"
#include "visitor/output_parameter_visitor.hpp"
#include "visitor/reset_visitor.hpp"

#include "init_rules/network_init.hpp"
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <ensmallen.hpp>

namespace mlpack {

/**
 * Implementation of a standard bidirectional recurrent neural network container.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename OutputLayerType = NegativeLogLikelihood<>,
  typename MergeLayerType = Concat<>,
  typename MergeOutputType = LogSoftMax<>,
  typename InitializationRuleType = RandomInitialization,
  typename... CustomLayers
>
class BRNN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = BRNN<OutputLayerType,
                           MergeLayerType,
                           MergeOutputType,
                           InitializationRuleType,
                           CustomLayers...>;

  /**
   * Create the BRNN object.
   *
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   * @param single Predict only the last element of the input sequence.
   * @param mergeLayer Merge layer to be used to evaluate the network.
   * @param outputLayer Output layer used to evaluate the network.
   * @param mergeOutput Output Merge layer to be used.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  BRNN(const size_t rho,
       const bool single = false,
       OutputLayerType outputLayer = OutputLayerType(),
       MergeLayerType* mergeLayer = new MergeLayerType(),
       MergeOutputType* mergeOutput = new MergeOutputType(),
       InitializationRuleType initializeRule = InitializationRuleType());

  ~BRNN();

  /**
   * Check if the optimizer has MaxIterations() parameter, if it does
   * then check if it's value is less than the number of datapoints
   * in the dataset.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param optimizer optimizer used in the training process.
   * @param samples Number of datapoints in the dataset.
   */
  template<typename OptimizerType>
  std::enable_if_t<
      HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>
  WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;

  /**
   * Check if the optimizer has MaxIterations() parameter, if it
   * doesn't then simply return from the function.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param optimizer optimizer used in the training process.
   * @param samples Number of datapoints in the dataset.
   */
  template<typename OptimizerType>
  std::enable_if_t<
      !HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>
  WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;

  /**
   * Train the bidirectional recurrent neural network on the given input data
   * using the given optimizer.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * The format of the data should be as follows:
   *  - each slice should correspond to a time step
   *  - each column should correspond to a data point
   *  - each row should correspond to a dimension
   * So, e.g., predictors(i, j, k) is the i'th dimension of the j'th data point
   * at time slice k.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   */
  template<typename OptimizerType>
  double Train(arma::cube predictors,
               arma::cube responses,
               OptimizerType& optimizer);

  /**
   * Train the bidirectional recurrent neural network on the given input data.
   * By default, the SGD optimization algorithm is used, but others can be specified
   * (such as ens::RMSprop).
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * The format of the data should be as follows:
   *  - each slice should correspond to a time step
   *  - each column should correspond to a data point
   *  - each row should correspond to a dimension
   * So, e.g., predictors(i, j, k) is the i'th dimension of the j'th data point
   * at time slice k.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   */
  template<typename OptimizerType = ens::StandardSGD>
  double Train(arma::cube predictors, arma::cube responses);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * output layer function.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * The format of the data should be as follows:
   *  - each slice should correspond to a time step
   *  - each column should correspond to a data point
   *  - each row should correspond to a dimension
   * So, e.g., predictors(i, j, k) is the i'th dimension of the j'th data point
   * at time slice k.  The responses will be in the same format.
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   * @param batchSize Number of points to predict at once.
   */
  void Predict(arma::cube predictors,
               arma::cube& results,
               const size_t batchSize = 256);

  /**
   * Evaluate the bidirectional recurrent neural network with the given
   * parameters. This function is usually called by the optimizer to train
   * the model.
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   * @param deterministic Whether or not to train or test the model. Note some
   *        layer act differently in training or testing mode.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t begin,
                  const size_t batchSize,
                  const bool deterministic);

  /**
   * Evaluate the bidirectional recurrent neural network with the given
   * parameters. This function is usually called by the optimizer to train
   * the model.  This just calls the other overload of Evaluate() with
   * deterministic = true.
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t begin,
                  const size_t batchSize);

  /**
   * Evaluate the bidirectional recurrent neural network with the given
   * parameters. This function is usually called by the optimizer to train
   * the model.  This just calls the other overload of Evaluate()
   * with deterministic = true.
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param gradient Matrix to output gradient into.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& parameters,
                              const size_t begin,
                              GradType& gradient,
                              const size_t batchSize);

  /**
   * Evaluate the gradient of the bidirectional recurrent neural network
   * with the given parameters, and with respect to only one point in
   * the dataset. This is useful for optimizers such as SGD, which require
   * a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param begin Index of the starting point to use for objective function
   *        gradient evaluation.
   * @param gradient Matrix to output gradient into.
   * @param batchSize Number of points to be processed as a batch for objective
   *        function gradient evaluation.
   */
  void Gradient(const arma::mat& parameters,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  /*
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args);

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes<CustomLayers...> layer);

  //! Return the number of separable functions. (number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  //! Return the maximum length of backpropagation through time.
  const size_t& Rho() const { return rho; }
  //! Modify the maximum length of backpropagation through time.
  size_t& Rho() { return rho; }

  //! Get the matrix of responses to the input data points.
  const arma::cube& Responses() const { return responses; }
  //! Modify the matrix of responses to the input data points.
  arma::cube& Responses() { return responses; }

  //! Get the matrix of data points (predictors).
  const arma::cube& Predictors() const { return predictors; }
  //! Modify the matrix of data points (predictors).
  arma::cube& Predictors() { return predictors; }

  /**
   * Reset the state of the network.  This ensures that all internally-held
   * gradients are set to 0, all memory cells are reset, and the parameters
   * matrix is the right size.
   */
  void Reset();

  /**
   * Reset the module information (weights/parameters).
   */
  void ResetParameters();

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  // Helper functions.
  /**
   * Reset the module status by setting the current deterministic parameter
   * for all modules that implement the Deterministic function.
   */
  void ResetDeterministic();

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Instantiated outputlayer used to evaluate the network.
  OutputLayerType outputLayer;

  //! Locally-stored merge Layer
  LayerTypes<CustomLayers...> mergeLayer;

  //! Locally-stored merge Layer
  LayerTypes<CustomLayers...> mergeOutput;

  //! Instantiated InitializationRule object for initializing the network
  //! parameter.
  InitializationRuleType initializeRule;

  //! The input size.
  size_t inputSize;

  //! The output size.
  size_t outputSize;

  //! The target size.
  size_t targetSize;

  //! Indicator if we already trained the model.
  bool reset;

    //! Only predict the last element of the input sequence.
  bool single;

  //! The matrix of data points (predictors).
  arma::cube predictors;

  //! The matrix of responses to the input data points.
  arma::cube responses;

  //! Matrix of (trained) parameters.
  arma::mat parameter;

  //! The number of separable functions (the number of predictor points).
  size_t numFunctions;

  //! The current error for the backward pass.
  arma::mat error;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! All output parameters for the backward pass (BBTT) for forward RNN.
  std::vector<arma::mat> forwardRNNOutputParameter;

  //! All output parameters for the backward pass (BBTT) for backward RNN.
  std::vector<arma::mat> backwardRNNOutputParameter;

  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;

  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored delete visitor.
  CopyVisitor<CustomLayers...> copyVisitor;

  //! The current evaluation mode (training or testing).
  bool deterministic;

  //! The current gradient for the gradient pass for forward RNN.
  arma::mat forwardGradient;

  //! The current gradient for the gradient pass for backward RNN.
  arma::mat backwardGradient;

  //! The total gradient from each gradient pass.
  arma::mat totalGradient;

  //! Forward RNN
  RNN<OutputLayerType, InitializationRuleType, CustomLayers...> forwardRNN;

  //! Backward RNN
  RNN<OutputLayerType, InitializationRuleType, CustomLayers...> backwardRNN;
}; // class BRNN

} // namespace mlpack

// Include implementation.
#include "brnn_impl.hpp"

#endif
