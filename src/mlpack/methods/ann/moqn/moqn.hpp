/**
 * @file methods/ann/moqn.hpp
 * @author Nanubala Gnana Sai
 *
 * Definition of the Multi-objective Q-network (MOQN) class, extends feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MORL_NETWORK_HPP
#define MLPACK_METHODS_ANN_MORL_NETWOKR_HPP

#include <mlpack/methods/ann/ffn.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of a standard feed forward network.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam CustomLayers Any set of custom layers that could be a part of the
 *         feed forward network.
 */
template<
  typename OutputLayerType = NegativeLogLikelihood<>,
  typename InitializationRuleType = RandomInitialization,
  typename... CustomLayers
>
class MOQN : public FFN<OutputLayerType, InitializationRuleType, CustomLayers...>
{
 public:
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
   * @tparam CallbackTypes Types of Callback Functions.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackTypes>
  double Train(arma::mat predictors,
               arma::mat responses,
               OptimizerType& optimizer,
               CallbackTypes&&... callbacks) = delete;

  /**
   * Train the feedforward network on the given input data. By default, the
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
  double Train(arma::mat predictors,
               arma::mat responses,
               CallbackTypes&&... callbacks) = delete;

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
   * Evaluate the feedforward network with the given predictors and responses.
   * This functions is usually used to monitor progress while training.
   *
   * @param predictors Input variables.
   * @param responses Target outputs for input variables.
   */
  template<typename PredictorsType, typename ResponsesType>
  double Evaluate(const PredictorsType& predictors,
                  const ResponsesType& responses) = delete;

  /**
   * Evaluate the feedforward network with the given parameters. This function
   * is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   */
  double Evaluate(const arma::mat& parameters) = delete;

   /**
   * Evaluate the feedforward network with the given parameters, but using only
   * a number of data points. This is useful for optimizers such as SGD, which
   * require a separable objective function.
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
                  const bool deterministic) = delete;

   /**
   * Evaluate the feedforward network with the given parameters, but using only
   * a number of data points. This is useful for optimizers such as SGD, which
   * require a separable objective function. This just calls the overload of
   * Evaluate() with deterministic = true.
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t begin,
                  const size_t batchSize) = delete;

  /**
   * Evaluate the feedforward network with the given parameters.
   * This function is usually called by the optimizer to train the model.
   * This just calls the overload of EvaluateWithGradient() with batchSize = 1.
   *
   * @param parameters Matrix model parameters.
   * @param gradient Matrix to output gradient into.
   */
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& parameters, GradType& gradient) = delete;

   /**
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
  template<typename GradType>
  double EvaluateWithGradient(const arma::mat& parameters,
                              const size_t begin,
                              GradType& gradient,
                              const size_t batchSize) = delete;

  /**
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
  void Gradient(const arma::mat& parameters,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize);

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
               const size_t end) = delete;

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
                  GradientsType& gradients) = delete;
  /**
   * Perform the backward pass for the EQL method.
   *
   * This method is intended to be used only for the EQL method. Homotopy loss calculation is handled
   * internally along with gradient calculation.
   *
   * @param inputs Inputs of current pass.
   * @param targets The training target.
   * @param extendedWeightSpace The extended preference vectors repository.
   * @param lambda The parameter controlling loss surface.
   * @param gradients Computed gradients.
   * @return Training error of the current pass.
   */
  template<typename PredictorsType,
           typename TargetsType,
           typename WeightsType,
           typename GradientsType>
  double Backward(const PredictorsType& inputs,
                  const TargetsType& targets,
                  const WeightsType& extendedWeightSpace,
                  double lambda,
                  GradientsType& gradients);
}; // class MOQN

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "morl_network_impl.hpp"

#endif
