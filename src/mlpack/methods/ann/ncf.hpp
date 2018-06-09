/**
 * @file ncf.hpp
 * @author Haritha Nair
 *
 * Definition of the NCFNetwork class, which implements feed forward neural
 * network for neural collaborative filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_NCF_HPP
#define MLPACK_METHODS_ANN_NCF_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/ffn.hpp>
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
 * @tparam Model The class type of user and item networks.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename Model,
  typename InitializationRuleType = RandomInitialization
>
class NCFNetwork
{
 public:
  //! Convenience typedef for the internal model construction.
  /**
   * Create the NCFNetwork object.
   *
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param userModel FFN model for user data.
   * @param itemModel FFN model for item data.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  NCFNetwork(Model& userModel, Model& itemModel,
      InitializationRuleType initializeRule = InitializationRuleType());

  /**
   * Train the network on the given input data using the given optimizer.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param user User's input training variables.
   * @param item Item's input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   */
  template<typename OptimizerType>
  void Train(arma::mat user,
             arma::mat item,
             arma::mat responses,
             OptimizerType& optimizer);

  /**
   * Perform the forward pass of the data in real batch mode.
   *
   * @param userInput The user input data.
   * @param itemInput The item input data.
   * @param results The predicted results.
   */
  void Forward(arma::mat userInput, arma::mat itemInput, arma::mat& results);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * output layer function.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param user User's input data.
   * @param item Item's input data.
   * @param results Matrix to put output predictions of responses into.
   */
  void Predict(arma::mat user, arma::mat item, arma::mat& results);

  /**
   * Evaluate the network with the given parameters. This function
   * is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   */
  double Evaluate(const arma::mat& parameters);

   /**
   * Evaluate the network with the given parameters, but using only one data
   * point. This is useful for optimizers such as SGD, which require a
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
                  const bool deterministic);

  /**
   * Evaluate the gradient of the network with the given parameters, and with
   * respect to only one point in the dataset. This is useful for optimizers
   * such as SGD, which require a separable objective function.
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

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  //! Get the matrix of responses to the input data points.
  const arma::mat& Responses() const { return responses; }
  //! Modify the matrix of responses to the input data points.
  arma::mat& Responses() { return responses; }

  //! Get the matrix of data points of user.
  const arma::mat& User() const { return user; }
  //! Modify the matrix of data points of user.
  arma::mat& User() { return user; }

  //! Get the matrix of data points of item.
  const arma::mat& Item() const { return item; }
  //! Modify the matrix of data points of item.
  arma::mat& Item() { return item; }

  //! Return the number of separable functions.
  size_t NumFunctions() const { return numFunctions; }

  /**
   * Reset the module infomration (weights/parameters).
   */
  void ResetParameters();

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  // Helper functions.
  /**
   * The Forward algorithm (part of the Forward-Backward algorithm).  Computes
   * forward probabilities for each module.
   *
   * @param userInput User's input data sequence to compute probabilities for.
   * @param itemInput Item's input data sequence to compute probabilities for.
   */
  void Forward(arma::mat&& userInput, arma::mat&& itemInput);

  /**
   * Prepare the network for the given data.
   * This function won't actually trigger training process.
   *
   * @param user User's input data variables.
   * @param item Item's input data variables.
   * @param responses Outputs results from input data variables.
   */
  void ResetData(arma::mat user, arma::mat item, arma::mat responses);

  /**
   * Reset the module status by setting the current deterministic parameter
   * for all modules that implement the Deterministic function.
   */
  void ResetDeterministic();

  /**
   * Reset the gradient for all modules that implement the Gradient function.
   */
  void ResetGradients(arma::mat& gradient);

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
  std::vector<LayerTypes<CustomLayers...> > network;

  //! The matrix of data points of user.
  arma::mat user;

  //! The matrix of data points of item.
  arma::mat item;

  //! The matrix of responses to the input data points.
  arma::mat responses;

  //! Matrix of (trained) parameters.
  arma::mat parameter;

  //! The number of separable functions.
  size_t numFunctions;

  //! The current error for the backward pass.
  arma::mat error;

  //! THe current input of the forward/backward pass.
  arma::mat curUserInput;

  //! THe current input of the forward/backward pass.
  arma::mat curItemInput;

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
  CopyVisitor<CustomLayers...> copyVisitor;
}; // class NCFNetwork

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "ncf_impl.hpp"

#endif
