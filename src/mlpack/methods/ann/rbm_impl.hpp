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

#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artificial neural networks. */ {


template<typename InitializationRuleType, typename RBMPolicy>
RBM<InitializationRuleType, RBMPolicy>::RBM(
    arma::Mat<ElemType> predictors,
    InitializationRuleType initializeRule,
    RBMPolicy rbmPolicy,
    const size_t numSteps,
    const size_t negSteps,
    const bool useMonitoringCost,
    const bool persistence):
    rbmPolicy(rbmPolicy),
    initializeRule(initializeRule),
    numSteps(numSteps),
    negSteps(negSteps),
    useMonitoringCost(useMonitoringCost),
    persistence(persistence),
    reset(false)
{
  numFunctions = predictors.n_cols;
  this->predictors = std::move(predictors);
}

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::Reset()
{
  size_t weight = rbmPolicy.Parameters().n_elem;

  positiveGradient.set_size(weight, 1);
  negativeGradient.set_size(weight, 1);
  negativeSamples.set_size(rbmPolicy.VisibleSize(), 1);
  tempNegativeGradient.set_size(weight, 1);

  parameter.zeros(weight, 1);
  positiveGradient.zeros();
  negativeGradient.zeros();
  tempNegativeGradient.zeros();
  initializeRule.Initialize(parameter, parameter.n_elem, 1);

  rbmPolicy.Parameters() = arma::Mat<ElemType>(parameter.memptr(), weight, 1,
      false, false);
  rbmPolicy.Reset();

  reset = true;
}

template<typename InitializationRuleType, typename RBMPolicy>
template<typename OptimizerType>
void RBM<InitializationRuleType, RBMPolicy>::Train(
    const arma::Mat<ElemType>& predictors,
    OptimizerType& optimizer)
{
  numFunctions = predictors.n_cols;
  this->predictors = std::move(predictors);

  if (!reset)
    Reset();

  // Train the model.
  Timer::Start("rbm_optimization");
  optimizer.Optimize(*this, parameter);
  Timer::Stop("rbm_optimization");
}

template<typename InitializationRuleType, typename RBMPolicy>
double RBM<InitializationRuleType, RBMPolicy>::
    FreeEnergy(arma::Mat<ElemType>&& input)
{
  return rbmPolicy.FreeEnergy(std::move(input));
}

template<typename InitializationRuleType, typename RBMPolicy>
double RBM<InitializationRuleType, RBMPolicy>::Evaluate(
    const arma::Mat<ElemType>& /* parameters*/,
    const size_t i)
{
  if (!useMonitoringCost)
  {
    Gibbs(std::move(predictors.col(i)), std::move(negativeSamples));
    return std::fabs(FreeEnergy(std::move(predictors.col(i))) -
        FreeEnergy(std::move(negativeSamples)));
  }
  else
  {
    if (persistence)
    {
      return rbmPolicy.Evaluate(predictors, i);
    }
    else
    {
      // Mean Squared Error
      rbmPolicy.SampleHidden(std::move(predictors.col(i)),
          std::move(hiddenReconstruction));
      rbmPolicy.SampleVisible(std::move(hiddenReconstruction),
          std::move(visibleReconstruction));
      return arma::accu(arma::pow(visibleReconstruction - predictors.col(i),
          2)) / hiddenReconstruction.n_rows;
    }
  }
}

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::SampleHidden(
    arma::Mat<ElemType>&& input,
    arma::Mat<ElemType>&& output)
{
  rbmPolicy.SampleHidden(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::SampleVisible(
    arma::Mat<ElemType>&& input,
    arma::Mat<ElemType>&& output)
{
  rbmPolicy.SampleVisible(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename RBMPolicy>
void RBM<InitializationRuleType, RBMPolicy>::Gibbs(
    arma::Mat<ElemType>&& input,
    arma::Mat<ElemType>&& output,
    size_t steps)
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
void RBM<InitializationRuleType, RBMPolicy>::Gradient(
    arma::Mat<ElemType>& /*parameters*/,
    const size_t input,
    arma::Mat<ElemType>& output)
{
  positiveGradient.zeros();
  // Collect the negative samples
  rbmPolicy.PositivePhase(std::move(predictors.col(input)),
      std::move(positiveGradient));

  negativeGradient.zeros();

  for (size_t i = 0; i < negSteps; i++)
  {
    Gibbs(std::move(predictors.col(input)), std::move(negativeSamples));
    rbmPolicy.NegativePhase(std::move(negativeSamples),
        std::move(tempNegativeGradient));

    negativeGradient += tempNegativeGradient;
  }
  output = ((negativeGradient / negSteps) - positiveGradient);
}

//! Serialize the model.
template<typename InitializationRuleType, typename RBMPolicy>
template<typename Archive>
void RBM<InitializationRuleType, RBMPolicy>::
    serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(parameter);
  ar & BOOST_SERIALIZATION_NVP(rbmPolicy);
  ar & BOOST_SERIALIZATION_NVP(state);
  ar & BOOST_SERIALIZATION_NVP(numFunctions);
  ar & BOOST_SERIALIZATION_NVP(numSteps);
  ar & BOOST_SERIALIZATION_NVP(negSteps);
  ar & BOOST_SERIALIZATION_NVP(useMonitoringCost);
  ar & BOOST_SERIALIZATION_NVP(persistence);
  // ar & data::CreateNVP(RBMPolicy::ElemType, "ElemType");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    size_t weight = rbmPolicy.Parameters().n_elem;
    positiveGradient.set_size(weight, 1);
    negativeGradient.set_size(weight, 1);
    negativeSamples.set_size(rbmPolicy.VisibleSize(), 1);
    tempNegativeGradient.set_size(weight, 1);
    positiveGradient.zeros();
    negativeGradient.zeros();
    tempNegativeGradient.zeros();
    rbmPolicy.Parameters() = arma::Mat<ElemType>(
        parameter.memptr(), weight, 1, false, false);
     rbmPolicy.Reset();
    reset = true;
  }
}

} // namespace ann
} // namespace mlpack
#endif // MLPACK_METHODS_ANN_RBM_IMPL_HPP
