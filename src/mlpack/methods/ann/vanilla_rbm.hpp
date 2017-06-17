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
#include "visitor/weight_set_visitor.hpp"
#include "visitor/forward_visitor.hpp"
#include "visitor/weight_size_visitor.hpp"
#include "activation_functions/softplus_function.hpp"
#include "init_rules/gaussian_init.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InitializationRuleType,
         typename VisibleLayerType,
         typename HiddenLayerType>
class RBM
{
 public:

  using NetworkType =RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>;

  /* 
   * Intalise all the parameters of the network
   * using the intialise rule. 
   *
   * @tparam: IntialiserType rule to intialise the parameters of the netwokr
   * @tparam: VisibleLayerType visible layer 
   * @tparam: HiddenLyaerType hidden layer
   */
  RBM(arma::mat predictors,
    InitializationRuleType initializeRule, 
    VisibleLayerType visible, 
    HiddenLayerType hidden);

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
  * This function calculates
  * the free energy of the model
  * @param: input data point 
  */
  double FreeEnergy(const arma::mat&& input);

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
  * @param negative_sample: stores the negative sample
  * @param num_step number of steps in gibbs sampling
  * @param persistnce start chain at input or the previous negative_sample
  */
  void Gibbs(arma::mat&& input, 
             arma::mat&& negative_sample, 
             size_t num_steps = 1, 
             bool persistence = false);


  /*
   * Calculates the gradients for the rbm network
   *
   * @param input index the visible layer/data point
   * @param neg_input stores the negative samples computed usign the gibbs 
   * @param k number of steps for gibbs sampling
   * @param persistence pcdk / not 
   * @param output store the gradients
   */
  void Gradient(const size_t input, 
                const size_t k, 
                const bool persistence, 
                arma::mat& output);

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; };
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; };
  
  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:

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
    arma::mat temp;
    ForwardVisible(std::move(input), std::move(temp)); 
    output = arma::vectorise(arma::join_cols(arma::join_cols(temp * input.t(), temp.t()), input.t()));
  };

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
  //! Locally-stored delete visitor.
  arma::mat state;
  //! Locally-stored reset variable
  bool reset;
  //! Locally-stored number of functions varaiable
  size_t numFunctions;

};
} // artificial neural network
} // mlpack
#include "vanilla_rbm_impl.hpp"
#endif