/**
 * @file vanilla_rbm_impl.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_RBM_IMPL_HPP
#define MLPACK_METHODS_ANN_RBM_IMPL_HPP

// In case it hasn't been included yet.
#include "rbm.hpp"

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>


#include "layer/layer.hpp"
#include "layer/base_layer.hpp"

#include "activation_functions/softplus_function.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::RBM(
    arma::mat predictors,
    InitializationRuleType initializeRule,
    VisibleLayerType visible,
    HiddenLayerType hidden,
    const size_t numSteps,
    const bool useMonitoringCost,
    const bool persistence):
    visible(visible),
    hidden(hidden),
    initializeRule(initializeRule),
    numSteps(numSteps),
    useMonitoringCost(useMonitoringCost),
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
  size_t numHidden = 0;
  size_t numVisible = 0;
  weight+= visible.Parameters().n_elem;
  parameter.set_size(weight, 1);
  parameter.zeros();
  initializeRule.Initialize(parameter, parameter.n_elem, 1);
  visible.Parameters() = arma::mat(parameter.memptr(), weight, 1, false, false);
  hidden.Parameters() = arma::mat(parameter.memptr(), weight, 1, false, false);

  visible.Reset();
  hidden.Reset();

  numVisible = visible.Bias().n_rows;
  numHidden = hidden.Bias().n_rows;

  positiveGradient.set_size(weight, 1);
  negativeGradient.set_size(weight, 1);

  // Variable for Negative grad wrt weights
  weightNegativeGrad = arma::mat(
      negativeGradient.memptr(),
      visible.Weight().n_rows,
      visible.Weight().n_cols, false, false);

  // Variable for Negative grad wrt HiddenBias
  hiddenBiasNegativeGrad = arma::mat(
      negativeGradient.memptr() + weightNegativeGrad.n_elem,
      numHidden, 1, false, false);

  // Variable for Negative grad wrt VisibleBias
  visibleBiasNegativeGrad = arma::mat(
      negativeGradient.memptr() + weightNegativeGrad.n_elem +
      hiddenBiasNegativeGrad.n_elem,
      numVisible, 1, false, false);

  // Variable for Positive grad wrt weights
  weightPositiveGrad = arma::mat(positiveGradient.memptr(),
    visible.Weight().n_rows,
    visible.Weight().n_cols, false, false);

  // Variable for Positive grad wrt hidden bias
  hiddenBiasPositiveGrad = arma::mat(
    positiveGradient.memptr() + weightPositiveGrad.n_elem,
    numHidden, 1, false, false);

  // Variable for Positive grad wrt visible bias
  visibleBiasPositiveGrad = arma::mat(
      positiveGradient.memptr() + weightPositiveGrad.n_elem +
      hiddenBiasPositiveGrad.n_elem,
      numVisible, 1, false, false);

  gibbsTemporary.set_size(numHidden, 1);
  negativeSamples.set_size(numVisible, 1);
  preActivation.set_size(numHidden, 1);
  corruptInput.set_size(numVisible, 1);

  reset = true;
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
template<typename OptimizerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Train(const arma::mat& predictors, OptimizerType& optimizer)
{
  numFunctions = predictors.n_cols;
  this->predictors = std::move(predictors);

  if (!reset)
  {
    Reset();
  }

  // Train the model.
  Timer::Start("rbm_optimization");
  optimizer.Optimize(*this, parameter);
  Timer::Stop("rbm_optimization");
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
double RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    FreeEnergy(arma::mat&& input)
{
  visible.ForwardPreActivation(std::move(input), std::move(preActivation));
  SoftplusFunction::Fn(preActivation, preActivation);
  return  -(arma::accu(preActivation) + arma::dot(input, visible.Bias()));
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
double RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Evaluate(const arma::mat& /* parameters*/, const size_t i)
{
  corruptInput = arma::round(predictors.col(i));
  if (!useMonitoringCost)
  {
    // Do not use unsafe col predictor as we change the input
    Gibbs(std::move(predictors.col(i)), std::move(negativeSamples));
    return (FreeEnergy(std::move(predictors.col(i))) -
        FreeEnergy(std::move(negativeSamples)));
  }
  else
  {
    if (persistence)
    {
      size_t idx = RandInt(0, predictors.n_rows);
      corruptInput.row(idx) = 1 - corruptInput.row(idx);
      return std::log(LogisticFunction::Fn(FreeEnergy(std::move(corruptInput)) -
          FreeEnergy(std::move(arma::round(predictors.col(i)))))) *
          predictors.n_rows;
    }
    else
    {
      // Cross entropy loss
      visible.Forward(std::move(predictors.col(i)),
        std::move(preActivation));
      preActivation = -preActivation;
      SoftplusFunction::Fn(preActivation, softOutput);
      softOutput = -softOutput;
      preActivation = -preActivation - 1;
      SoftplusFunction::Fn(preActivation, softOutput);
      softOutput = -softOutput;
      return arma::accu(
          predictors.col(i) * softOutput.t() + (1 - predictors.col(i)) *
          preActivation.t());
    }
  }
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
    Gibbs(arma::mat&& input, arma::mat&& output, size_t steps)
{
  steps = (steps == SIZE_MAX) ? this-> numSteps: steps;

  if (persistence && !state.is_empty())
  {
    SampleHidden(std::move(state), std::move(gibbsTemporary));
    SampleVisible(std::move(gibbsTemporary), std::move(output));
  }
  else
  {
    SampleHidden(std::move(input), std::move(gibbsTemporary));
    SampleVisible(std::move(gibbsTemporary), std::move(output));
  }

  for (size_t j = 1; j < steps; j++)
  {
    SampleHidden(std::move(output), std::move(gibbsTemporary));
    SampleVisible(std::move(gibbsTemporary), std::move(output));
  }
  if (persistence)
    state = output;
}

template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Gradient(arma::mat& /*parameters*/, const size_t input, arma::mat& output)
{
  visible.Forward(std::move(predictors.col(input)),
      std::move(hiddenBiasPositiveGrad));
  weightPositiveGrad = hiddenBiasPositiveGrad * predictors.col(input).t();
  visibleBiasPositiveGrad = predictors.col(input);

  // Collect the negative samples
  Gibbs(std::move(predictors.col(input)), std::move(negativeSamples));

  // Collect the negative gradients
  visible.Forward(std::move(negativeSamples),
      std::move(hiddenBiasNegativeGrad));
  weightNegativeGrad = hiddenBiasNegativeGrad * negativeSamples.t();
  visibleBiasNegativeGrad = negativeSamples;

  output = (negativeGradient - positiveGradient);
}

//! Serialize the model.
template<typename InitializationRuleType, typename VisibleLayerType,
    typename HiddenLayerType>
template<typename Archive>
void RBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::
    Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(visible, "visible");
  ar & data::CreateNVP(hidden, "hidden");

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
