/**
 * @file vanilla_rbm.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VANILLA_RBM_HPP
#define MLPACK_METHODS_ANN_VANILLA_RBM_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

#include "layer/layer.hpp"
#include "layer/base_layer.hpp"

#include "activation_functions/softplus_function.hpp"
#include "init_rules/gaussian_init.hpp"
#include "init_rules/random_init.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InitializationRuleType,
    typename VisibleLayerType,
    typename HiddenLayerType>
class RBM
{
 public:
  using NetworkType = RBM<InitializationRuleType, VisibleLayerType,
      HiddenLayerType>;

  /* 
   * Intalise all the parameters of the network
   * using the intialise rule. 
   *
   * @tparam: IntialiserType rule to intialise the parameters of the network
   * @tparam: VisibleLayerType visible layer 
   * @tparam: HiddenLyaerType hidden layer
   */
  RBM(arma::mat predictors, InitializationRuleType initializeRule,
      VisibleLayerType visible,
      HiddenLayerType hidden,
      const size_t numSteps = 1,
      const bool useMonitoringCost = true,
      const bool persistence = false);

  // Reset the network
  void Reset();

  /* 
   * Train the netwrok using the Opitimzer with given set of args.
   * the optimiser sets the paratmeters of the network for providing
   * most likely parameters given the inputs
   * @param: predictors data points
   * @param: optimizer Optimizer type
   * @param: Args arguments for the optimizers
   */
  template<template <typename> class Optimizer>
  void Train(const arma::mat& predictors, Optimizer<NetworkType>& optimizer);

 /**
  * Evaluate the rbm network with the given parameters. This function
  * is usually called by the optimizer to train the model.
  * The function computes the pseudo likelihood
  *
  * @param parameters Matrix model parameters.
  * @param i Index of point to use for objective function evaluation.
  * 
  */
  double Evaluate(const arma::mat& parameters, const size_t i);

  /**
  * Monitor Cost this function is needed for checking
  * the progress of the training. Cross-Entropy is 
  * needed when peristence is false and pseudo-likelihood
  * is needed when persistence is true
  *
  * @param i Index of point to use for objective function evaluation.
  * 
  */
  double MonitoringCost(const size_t i);

 /** 
  * This function calculates
  * the free energy of the model
  * @param: input data point 
  */
  double FreeEnergy(arma::mat&& input);

 /*
  * This functions samples the hidden
  * layer given the visible layer
  *
  * @param input visible layer input
  * @param output the sampled hidden layer
  */
  void SampleHidden(arma::mat&& input, arma::mat&& output);

  /*
  * This functions samples the visble
  * layer given the hidden layer
  *
  * @param input hidden layer
  * @param output the sampled visible layer 
  */
  void SampleVisible(arma::mat&& input, arma::mat&& output);

 /*
  * This function does the k-step
  * gibbs sampling.
  *
  * @param input: input to the gibbs function
  * @param output: stores the negative sample
  */
  void Gibbs(arma::mat&& input, arma::mat&& output, size_t steps = SIZE_MAX);

  /*
   * Calculates the gradients for the rbm network
   *
   * @param input index the visible layer/data point
   * @param neg_input stores the negative samples computed usign the gibbs 
   * @param k number of steps for gibbs sampling
   * @param persistence pcdk / not 
   * @param output store the gradients
   */
  void Gradient(const size_t input, arma::mat& output);

  /* 
   * ForwardVisible layer compute the forward
   * activations given the visible layer
   *
   * @param input the visible layer
   * @param output the acitvation function
   */
  void ForwardVisible(arma::mat&& input, arma::mat&& output)
  {
    visible.Forward(std::move(input), std::move(output));
  };

  /* 
   * ForwardHidden layer compute the forward
   * activations given the hidden layer
   *
   * @param input the visible layer
   * @param output the acitvation function
   */
  void ForwardHidden(arma::mat&& input, arma::mat&& output)
  {
    hidden.Forward(std::move(input), std::move(output));
  };

  /*
   * Helper function for Gradient
   * calculates the gradients for both
   * positive and negative samples.
   */
  void CalcGradient(arma::mat&& input, arma::mat&& output)
  {
    ForwardVisible(std::move(input), std::move(inputForward));
    output = inputForward * input.t();
    // Weights, hidden bias, visible bias
    output = arma::join_cols(arma::join_cols(arma::vectorise(output),
        inputForward), input);
  };

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumSteps() const { return numSteps; }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  VisibleLayerType& VisibleLayer() { return visible; }
  HiddenLayerType& HiddenLayer() { return hidden; }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:

  // Parameter weights of the network
  arma::mat parameter;
  // Visible layer
  VisibleLayerType visible;
  // Hidden Layer
  HiddenLayerType hidden;
  // Sigmoid Layer
  LayerTypes sigmoid;
  // ResetVisitor
  ResetVisitor resetVisitor;
  // DeleteVistor
  DeleteVisitor deleteVisitor;
  //! The matrix of data points (predictors).
  arma::mat predictors;
  // Samples
  arma::mat samples;
  // Intialiser
  InitializationRuleType initializeRule;
  // Softplus function
  SoftplusFunction softplus;
  //! Locally-stored state of the persistent cdk.
  arma::mat state;
  //! Locally-stored number of functions varaiable
  size_t numFunctions;
  //! Locally-stored number of steps in gibbs sampling
  const size_t numSteps;
  //! Locally-stored monitoring cost
  const bool useMonitoringCost;
  //! Locally-stored persistent cd-k or not
  const bool persistence;
  //! Locally-stored reset variable
  bool reset;
  //! Locally-stored Forward output variable for positive phase
  arma::mat inputForward;

  arma::mat negativeGradient;
  arma::mat positiveGradient;
  arma::mat weightNegativeGrad;
  arma::mat hiddenBiasNegativeGrad;
  arma::mat visibleBiasNegativeGrad;

  arma::mat weightPositiveGrad;
  arma::mat hiddenBiasPositiveGrad;
  arma::mat visibleBiasPositiveGrad;

  arma::mat gibbsTemporary;
  arma::mat negativeSamples;
  arma::mat activation;
  arma::mat preActivation;
  arma::mat corruptInput;
};
} // namespace ann
} // namespace mlpack
#include "rbm_impl.hpp"
#endif
