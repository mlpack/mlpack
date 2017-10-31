/**
 * @file ffn.hpp
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

#include <mlpack/prereqs.hpp>

#include "visitor/delete_visitor.hpp"
#include "visitor/delta_visitor.hpp"
#include "visitor/output_height_visitor.hpp"
#include "visitor/output_parameter_visitor.hpp"
#include "visitor/output_width_visitor.hpp"
#include "visitor/reset_visitor.hpp"
#include "visitor/weight_size_visitor.hpp"
#include "visitor/copy_visitor.hpp"

#include "init_rules/network_init.hpp"

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of a standard feed forward network.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename OutputLayerType = NegativeLogLikelihood<>,
  typename InitializationRuleType = RandomInitialization
>
class FFN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = FFN<OutputLayerType, InitializationRuleType>;

  /**
   * Create the FFN object with the given predictors and responses set (this is
   * the set that is used to train the network).
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
  FFN(const FFN&);

  //! Move constructor.
  FFN(FFN&&);

  //! Copy/move assignment operator.
  FFN& operator = (FFN);

  /**
   * Create the FFN object with the given predictors and responses set (this is
   * the set that is used to train the network).
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  FFN(arma::mat predictors,
      arma::mat responses,
      OutputLayerType outputLayer = OutputLayerType(),
      InitializationRuleType initializeRule = InitializationRuleType());

  //! Destructor to release allocated memory.
  ~FFN();

  /**
   * Train the feedforward network on the given input data using the given
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
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   */
  template<typename OptimizerType>
  void Train(arma::mat predictors,
             arma::mat responses,
             OptimizerType& optimizer);

  /**
   * Train the feedforward network on the given input data. By default, the
   * RMSProp optimization algorithm is used, but others can be specified
   * (such as mlpack::optimization::SGD).
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
   * @param responses Outputs results from input training variables.
   */
  template<typename OptimizerType = mlpack::optimization::RMSProp>
  void Train(arma::mat predictors, arma::mat responses);

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
   */
  void Predict(arma::mat predictors, arma::mat& results);

  /**
   * Evaluate the feedforward network with the given parameters. This function
   * is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   * @param deterministic Whether or not to train or test the model. Note some
   *        layer act differently in training or testing mode.
   */
  double Evaluate(const arma::mat& parameters);

   /**
   * Evaluate the feedforward network with the given parameters, but using only
   * one data point. This is useful for optimizers such as SGD, which require a
   * separable objective function.
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
                  const bool deterministic = true);

  /**
   * Evaluate the gradient of the feedforward network with the given parameters,
   * and with respect to only one point in the dataset. This is useful for
   * optimizers such as SGD, which require a separable objective function.
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
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes layer) { network.push_back(layer); }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  /**
   * Reset the module infomration (weights/parameters).
   */
  void ResetParameters();

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

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
  void Forward(arma::mat inputs, arma::mat& results);

  /**
   * Perform the backward pass of the data in real batch mode.
   *
   * Forward and Backward should be used as a pair, and they are designed mainly
   * for advanced users. User should try to use Predict and Train unless those
   * two functions can't satisfy some special requirements.
   *
   * @param targets The training target.
   * @param gradients Computed gradients.
   * @return Training error of the current pass.
   */
  double Backward(arma::mat targets, arma::mat& gradients);

 private:
  // Helper functions.
  /**
   * The Forward algorithm (part of the Forward-Backward algorithm).  Computes
   * forward probabilities for each module.
   *
   * @param input Data sequence to compute probabilities for.
   */
  void Forward(arma::mat&& input);

  /**
   * Prepare the network for the given data.
   * This function won't actually trigger training process.
   *
   * @param predictors Input data variables.
   * @param responses Outputs results from input data variables.
   */
  void ResetData(arma::mat predictors, arma::mat responses);

  /**
   * The Backward algorithm (part of the Forward-Backward algorithm). Computes
   * backward pass for module.
   */
  void Backward();

  /**
   * Iterate through all layer modules and update the the gradient using the
   * layer defined optimizer.
   */
  void Gradient(arma::mat&& input);

  /**
   * Reset the module status by setting the current deterministic parameter
   * for all modules that implement the Deterministic function.
   */
  void ResetDeterministic();

  /**
   * Reset the gradient for all modules that implement the Gradient function.
   */
  void ResetGradients(arma::mat& gradient);

  /**
   * Swap the content of this network with given network.
   *
   * @param network Desired source network.
   */
  void Swap(FFN& network);

  //! Instantiated outputlayer used to evaluate the network.
  OutputLayerType outputLayer;

  //! Instantiated InitializationRule object for initializing the network
  //! parameter.
  InitializationRuleType initializeRule;

  //! The input width.
  size_t width;

  //! The input height.
  size_t height;

  //! Indicator if we already trained the model.
  bool reset;

  //! Locally-stored model modules.
  std::vector<LayerTypes> network;

  //! The matrix of data points (predictors).
  arma::mat predictors;

  //! The matrix of responses to the input data points.
  arma::mat responses;

  //! Matrix of (trained) parameters.
  arma::mat parameter;

  //! The number of separable functions (the number of predictor points).
  size_t numFunctions;

  //! The current error for the backward pass.
  arma::mat error;

  //! THe current input of the forward/backward pass.
  arma::mat currentInput;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;

  //! Locally-stored output width visitor.
  OutputWidthVisitor outputWidthVisitor;

  //! Locally-stored output height visitor.
  OutputHeightVisitor outputHeightVisitor;

  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! The current evaluation mode (training or testing).
  bool deterministic;

  //! Locally-stored delta object.
  arma::mat delta;

  //! Locally-stored input parameter object.
  arma::mat inputParameter;

  //! Locally-stored output parameter object.
  arma::mat outputParameter;

  //! Locally-stored gradient parameter.
  arma::mat gradient;

  //! Locally-stored copy visitor
  CopyVisitor copyVisitor;
}; // class FFN

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "ffn_impl.hpp"

#endif
