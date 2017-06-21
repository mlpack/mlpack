/**
 * @file vanilla_rbm_impl.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_VANILLA_RBM_IMPL_HPP
#define MLPACK_METHODS_ANN_VANILLA_RBM_IMPL_HPP

// In case it hasn't been included yet.
#include "vanilla_rbm.hpp"

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>


#include "layer/layer.hpp"
#include "layer/base_layer.hpp"

#include "init_rules/gaussian_init.hpp"
#include "activation_functions/softplus_function.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


// Intialise a linear layer and sigmoid layer
template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::RBM(
    arma::mat predictors,
    InitializationRuleType initializeRule,
    VisibleLayerType visible,
    HiddenLayerType hidden,
    const size_t numSteps,
    const bool persistence):
    visible(visible),
    hidden(hidden),
    initializeRule(initializeRule),
    numSteps(numSteps),
    persistence(persistence),
    reset(false)
{
  numFunctions = predictors.n_cols;
  this->predictors = std::move(predictors);
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::Reset()
{
  // Since we are dealing with a static stucture no need for visitor
  // Reset the parameters of all the layers
  size_t weight = 0;
  weight+= visible.Parameters().n_elem;
  parameter.set_size(weight, 1);
  initializeRule.Initialize(parameter, parameter.n_elem, 1);
  visible.Parameters() = arma::mat(parameter.memptr(), weight, 1, false, false);
  hidden.Parameters() = arma::mat(parameter.memptr(), weight, 1, false, false);

  visible.Reset();
  hidden.Reset();
  reset = true;
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
template<template<typename> class Optimizer>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Train(const arma::mat& predictors, Optimizer<NetworkType>& optimizer)
{
  numFunctions = predictors.n_cols;
  this->predictors = std::move(predictors);

  if (!reset)
  {
    Reset();
  }

  // Train the model.
  Timer::Start("rbm_optimization");
  optimizer.Optimize(parameter);
  Timer::Stop("rbm_optimization");
}

 /* 
  * This function calculates
  * the free energy of the model
  */
template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
double RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    FreeEnergy(arma::mat&& input)
{
  arma::mat output;
  visible.Forward(std::move(input), std::move(output));
  SoftplusFunction::Fn(output, output);
  return  -arma::accu(output) -arma::dot(input, visible.Bias());
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
double RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Evaluate(const arma::mat& /* parameters*/, const size_t i)
{
  
    double freeEnergy, freeEnergyChain;
    arma::mat negativeSample;
    arma::vec currentInput = predictors.unsafe_col(i);
    // Do not use unsafe col predictor as we change the input
    freeEnergy = FreeEnergy(std::move(currentInput));
    Gibbs(currentInput, std::move(negativeSample));
    freeEnergyChain = FreeEnergy(std::move(negativeSample));

    return arma::mean(freeEnergy) - arma::mean(freeEnergyChain);
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    SampleHidden(arma::mat&& input, arma::mat&& output)
{
  visible.Sample(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    SampleVisible(arma::mat&& input, arma::mat&& output)
{
  hidden.Sample(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Gibbs(arma::mat input, arma::mat&& output)
{
  if (persistence && !state.is_empty())
    input = state;

  for (size_t j = 0; j < numSteps; j++)
  {
    // Use probabilties for updation till the last step(section 3 hinton)
    SampleHidden(std::move(input), std::move(output));
    SampleVisible(std::move(output), std::move(input));
  }
  output = input;
  state = input;
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Gradient(const size_t input, arma::mat& output)
{
  arma::mat positiveGradient, negativeGradient, negativeSamples;
  // Collect the positive gradients
  CalcGradient(std::move(predictors.col(input)), std::move(positiveGradient));

  // Collect the negative samples
  Gibbs(std::move(predictors.col(input)), std::move(negativeSamples));

  // Collect the negative gradients
  CalcGradient(std::move(negativeSamples), std::move(negativeGradient));

  output = positiveGradient - negativeGradient;
}

//! Serialize the model.
template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
template<typename Archive>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    reset = false;

    size_t weight = 0;
    weight+= visible.Parameters().n_elem;
    visible.Parameters() = arma::mat(parameter.memptr(), weight, 1, false,
       false);
    hidden.Parameters() = arma::mat(parameter.memptr(), weight, 1, false,
        false);

    visible.Reset();
    hidden.Reset();
  }
}
} // namespace ann
} // namespace mlpack
#endif
