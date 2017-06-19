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
    arma::mat& predictors,
    InitializationRuleType initializeRule,
    VisibleLayerType visible,
    HiddenLayerType hidden):
    visible(visible),
    hidden(hidden),
    initializeRule(initializeRule),
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
    FreeEnergy(const arma::mat&& input)
{
  arma::mat output;
  visible.Forward(std::move(input), std::move(output));
  SoftplusFunction::Fn(output, output);
  return  -arma::dot(input, visible.Bias()) - arma::accu(output);
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
double RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Evaluate(const arma::mat& /* parameters*/, arma::vec& orderFunction)
{
  double freeEnergy, freeEnergyCorrupted;
  arma::Col<double> output(orderFunction.n_elem);
  size_t corruptIdx = math::RandInt(0, visible.Bias().n_elem);
  for (auto i : orderFunction)
  {
    // Do not use unsafe col predictor as we change the input
    arma::vec currentInput = predictors.col(i);
    freeEnergy = FreeEnergy(std::move(currentInput));
    currentInput(corruptIdx) =  1 - predictors(corruptIdx);
    freeEnergyCorrupted = FreeEnergy(std::move(currentInput));
    output(i) = SoftplusFunction::Fn(freeEnergyCorrupted - freeEnergy + 1);
  }
  // Concerned here for neumerical stability.
  return arma::accu(output) * visible.Bias().n_elem;
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
    Gibbs(arma::mat&& input,
    arma::mat&& negativeSamples,
    size_t numSteps,
    bool persistence)
{
  arma::mat temp, temp1;
  temp = input;

  if (persistence && !state.is_empty())
    input = state;

  for (size_t j = 0; j < numSteps - 1; j++)
  {
    // Use probabilties for updation till the last step(section 3 hinton)
    ForwardVisible(std::move(temp), std::move(temp1));
    ForwardHidden(std::move(temp1), std::move(temp));
  }

  // Finall samples should use the sampling
  SampleHidden(std::move(temp), std::move(temp1));
  SampleVisible(std::move(temp1), std::move(temp));

  state = temp;
  negativeSamples = state;
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Gradient(const size_t input,
        const size_t k,
        const bool persistence,
        arma::mat& output)
{
  arma::mat positiveGradient, negativeGradient, negativeSamples;
  // Collect the positive gradients
  CalcGradient(std::move(predictors.col(input)), std::move(positiveGradient));

  // Collect the negative samples
  Gibbs(std::move(predictors.col(input)), std::move(negativeSamples), k,
      persistence);

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
  // Todo: Save other parameters
  ar & data::CreateNVP(parameter, "parameter");
}
} // namespace ann
} // namespace mlpack
#endif
