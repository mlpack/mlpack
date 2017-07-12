/**
 * @file rbm_impl.hpp
 * @author Kris Singh
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


#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>

#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

using namespace mlpack;
using namespace mlpack::ann;

namespace mlpack {
namespace rbm /** Restricted Boltzmann Machine. */ {


template<typename InitializationRuleType, typename RBMPolicy>
RBM<InitializationRuleType, RBMPolicy>::RBM(
    arma::mat predictors,
    InitializationRuleType initializeRule,
    RBMPolicy rbmPolicy,
    const size_t numSteps,
    const size_t mSteps,
    const bool useMonitoringCost,
    const bool persistence):
    rbmPolicy(rbmPolicy),
    initializeRule(initializeRule),
    numSteps(numSteps),
    mSteps(mSteps),
    useMonitoringCost(useMonitoringCost),
    persistence(persistence),
    reset(false)
{
  numFunctions = predictors.n_cols;
  this->predictors = std::move(predictors);
}

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>
    ::Reset()
{
  size_t weight = 0;
  weight+= rbmPolicy.Parameters().n_elem;
  positiveGradient.set_size(weight, 1);
  negativeGradient.set_size(weight, 1);
  parameter.set_size(weight, 1);
  parameter.zeros();
  initializeRule.Initialize(parameter, parameter.n_elem, 1);
  rbmPolicy.Parameters() = arma::mat(parameter.memptr(), weight, 1, false,
      false);
  rbmPolicy.PositiveGradient() = arma::mat(positiveGradient.memptr(), weight, 1,
      false, false);
  rbmPolicy.NegativeGradient() = arma::mat(negativeGradient.memptr(), weight, 1,
      false, false);

  rbmPolicy.Reset();

  gibbsTemporary.set_size(rbmPolicy.VisibleLayer().InSize(), 1);
  negativeSamples.set_size(rbmPolicy.VisibleLayer().InSize(), 1);
  corruptInput.set_size(rbmPolicy.VisibleLayer().InSize(), 1);
  reset = true;
}

template<typename InitializationRuleType, typename RBMPolicy>
template<typename OptimizerType>
void RBM<InitializationRuleType, RBMPolicy>::
    Train(const arma::mat& predictors, OptimizerType& optimizer)
{
  numFunctions = predictors.n_cols;
  this->predictors = std::move(predictors);

  if (!reset)
  {
    std::cout << "Train Reset" << std::endl;
    Reset();
  }

  // Train the model.
  Timer::Start("rbm_optimization");
  optimizer.Optimize(*this, parameter);
  Timer::Stop("rbm_optimization");
}

template<typename InitializationRuleType, typename RBMPolicy>
double RBM<InitializationRuleType, RBMPolicy>
    ::FreeEnergy(arma::mat&& input)
{
  return rbmPolicy.FreeEnergy(std::move(input));
}

template<typename InitializationRuleType, typename RBMPolicy>
double RBM<InitializationRuleType, RBMPolicy>
    ::Evaluate(const arma::mat& /* parameters*/, const size_t i)
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
      rbmPolicy.VisibleLayer().Forward(std::move(predictors.col(i)),
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

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::
    SampleHidden(arma::mat&& input, arma::mat&& output)
{
  rbmPolicy.SampleHidden(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::
    SampleVisible(arma::mat&& input, arma::mat&& output)
{
  rbmPolicy.SampleVisible(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::
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

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::
    Gradient(arma::mat& /*parameters*/, const size_t input, arma::mat& output)
{
  // Collect the negative samples
  rbmPolicy.PositivePhase(std::move(predictors.col(input)));
  negativeSamples.set_size(predictors.n_rows, mSteps);
  for (size_t i = 0; i < mSteps; i++)
    Gibbs(std::move(predictors.col(input)), std::move(negativeSamples.col(i)));
  rbmPolicy.NegativePhase(std::move(negativeSamples));
  output = (negativeGradient - positiveGradient);
}

} // namespace rbm
} // namespace mlpack
#endif
