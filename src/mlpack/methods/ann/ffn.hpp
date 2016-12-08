/**
 * @file ffn.hpp
 * @author Marcus Edel
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

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/layer_visitor.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of a standard feed forward network.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */

// NguyenWidrowInitialization
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
   * the set that is used to train the network) and the given optimizer.
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  FFN(OutputLayerType&& outputLayer = OutputLayerType(),
      InitializationRuleType initializeRule = InitializationRuleType());

  /**
   * Create the FFN object with the given predictors and responses set (this is
   * the set that is used to train the network) and the given optimizer.
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  FFN(const arma::mat& predictors,
      const arma::mat& responses,
      OutputLayerType&& outputLayer = OutputLayerType(),
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
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   */
  template<
      template<typename> class OptimizerType = mlpack::optimization::RMSprop
  >
  void Train(const arma::mat& predictors,
             const arma::mat& responses,
             OptimizerType<NetworkType>& optimizer);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * output layer function.
   *
   * @param predictors Input predictors.
   * @param responses Matrix to put output predictions of responses into.
   */
  void Predict(arma::mat& predictors, arma::mat& responses);

  /**
   * Evaluate the feedforward network with the given parameters. This function
   * is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   * @param i Index of point to use for objective function evaluation.
   * @param deterministic Whether or not to train or test the model. Note some
   *        layer act differently in training or testing mode.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const bool deterministic = true);

  /**
   * Evaluate the gradient of the feedforward network with the given parameters,
   * and with respect to only one point in the dataset. This is useful for
   * optimizers such as SGD, which require a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param i Index of points to use for objective function gradient evaluation.
   * @param gradient Matrix to output gradient into.
   */
  void Gradient(const arma::mat& parameters,
                const size_t i,
                arma::mat& gradient);

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

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

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
   * The Backward algorithm (part of the Forward-Backward algorithm). Computes
   * backward pass for module.
   */
  void Backward();

  /**
   * Iterate through all layer modules and update the the gradient using the
   * layer defined optimizer.
   */
  void Gradient();

  /**
   * Reset the module infomration (weights/parameters).
   */
  void ResetParameters();

  /**
   * Reset the module status by setting the current deterministic parameter
   * for all modules that implement the Deterministic function.
   */
  void ResetDeterministic();

  /**
   * Reset the gradient for all modules that implement the Gradient function.
   */
  void ResetGradients(arma::mat& gradient);

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

  //! THe current target of the forward/backward pass.
  arma::mat currentTarget;

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
}; // class FFN

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "ffn_impl.hpp"

#endif
