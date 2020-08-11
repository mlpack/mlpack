/**
 * @file methods/ann/rbm/rbm_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_RBM_IMPL_HPP
#define MLPACK_METHODS_ANN_RBM_RBM_IMPL_HPP

// In case it hasn't been included yet.
#include "rbm.hpp"

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

namespace mlpack {
namespace ann /** Artificial neural networks. */ {

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
RBM<InitializationRuleType, DataType, PolicyType>::RBM(
    arma::Mat<ElemType> predictors,
    InitializationRuleType initializeRule,
    const size_t visibleSize,
    const size_t hiddenSize,
    const size_t batchSize,
    const size_t numSteps,
    const size_t negSteps,
    const size_t poolSize,
    const ElemType slabPenalty,
    const ElemType radius,
    const bool persistence) :
    predictors(std::move(predictors)),
    initializeRule(initializeRule),
    visibleSize(visibleSize),
    hiddenSize(hiddenSize),
    batchSize(batchSize),
    numSteps(numSteps),
    negSteps(negSteps),
    poolSize(poolSize),
    steps(0),
    slabPenalty(slabPenalty),
    radius(2 * radius),
    persistence(persistence),
    reset(false)
{
  numFunctions = this->predictors.n_cols;
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
RBM<InitializationRuleType, DataType, PolicyType>::Reset()
{
  size_t shape = (visibleSize * hiddenSize) + visibleSize + hiddenSize;

  parameter.set_size(shape, 1);
  positiveGradient.set_size(shape, 1);
  negativeGradient.set_size(shape, 1);
  tempNegativeGradient.set_size(shape, 1);
  negativeSamples.set_size(visibleSize, batchSize);

  weight = arma::Cube<ElemType>(parameter.memptr(), hiddenSize, visibleSize, 1,
      false, false);
  hiddenBias = DataType(parameter.memptr() + weight.n_elem,
      hiddenSize, 1, false, false);
  visibleBias = DataType(parameter.memptr() + weight.n_elem +
      hiddenBias.n_elem, visibleSize, 1, false, false);

  parameter.zeros();
  positiveGradient.zeros();
  negativeGradient.zeros();
  tempNegativeGradient.zeros();
  initializeRule.Initialize(parameter, parameter.n_elem, 1);

  reset = true;
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename OptimizerType, typename... CallbackType>
double RBM<InitializationRuleType, DataType, PolicyType>::Train(
    OptimizerType& optimizer, CallbackType&&... callbacks)
{
  if (!reset)
  {
    Reset();
  }

  return optimizer.Optimize(*this, parameter, callbacks...);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, double>::type
RBM<InitializationRuleType, DataType, PolicyType>::FreeEnergy(
    const arma::Mat<ElemType>& input)
{
  preActivation = (weight.slice(0) * input);
  preActivation.each_col() += hiddenBias;
  return -(arma::accu(arma::log(1 + arma::trunc_exp(preActivation))) +
      arma::dot(input, arma::repmat(visibleBias, 1, input.n_cols)));
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
RBM<InitializationRuleType, DataType, PolicyType>::Phase(
    const InputType& input,
    DataType& gradient)
{
  arma::Cube<ElemType> weightGrad = arma::Cube<ElemType>(gradient.memptr(),
      hiddenSize, visibleSize, 1, false, false);

  DataType hiddenBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  HiddenMean(input, hiddenBiasGrad);
  weightGrad.slice(0) = hiddenBiasGrad * input.t();
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
double RBM<InitializationRuleType, DataType, PolicyType>::Evaluate(
    const arma::Mat<ElemType>& /* parameters*/,
    const size_t i,
    const size_t batchSize)
{
  Gibbs(predictors.cols(i, i + batchSize - 1),
      negativeSamples);
  return std::fabs(FreeEnergy(predictors.cols(i,
      i + batchSize - 1)) - FreeEnergy(negativeSamples));
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
RBM<InitializationRuleType, DataType, PolicyType>::SampleHidden(
    const arma::Mat<ElemType>& input,
    arma::Mat<ElemType>& output)
{
  HiddenMean(input, output);

  for (size_t i = 0; i < output.n_elem; ++i)
  {
    output(i) = math::RandBernoulli(output(i));
  }
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
RBM<InitializationRuleType, DataType, PolicyType>::SampleVisible(
    arma::Mat<ElemType>& input,
    arma::Mat<ElemType>& output)
{
  VisibleMean(input, output);

  for (size_t i = 0; i < output.n_elem; ++i)
  {
    output(i) = math::RandBernoulli(output(i));
  }
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
RBM<InitializationRuleType, DataType, PolicyType>::VisibleMean(
    InputType& input,
    DataType& output)
{
  output = weight.slice(0).t() * input;
  output.each_col() += visibleBias;
  LogisticFunction::Fn(output, output);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
RBM<InitializationRuleType, DataType, PolicyType>::HiddenMean(
    const InputType& input,
    DataType& output)
{
  output = weight.slice(0) * input;
  output.each_col() += hiddenBias;
  LogisticFunction::Fn(output, output);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
void RBM<InitializationRuleType, DataType, PolicyType>::Gibbs(
    const arma::Mat<ElemType>& input,
    arma::Mat<ElemType>& output,
    const size_t steps)
{
  this->steps = (steps == SIZE_MAX) ? this->numSteps : steps;

  if (persistence && !state.is_empty())
  {
    SampleHidden(state, gibbsTemporary);
    SampleVisible(gibbsTemporary, output);
  }
  else
  {
    SampleHidden(input, gibbsTemporary);
    SampleVisible(gibbsTemporary, output);
  }

  for (size_t j = 1; j < this->steps; ++j)
  {
    SampleHidden(output, gibbsTemporary);
    SampleVisible(gibbsTemporary, output);
  }
  if (persistence)
  {
    state = output;
  }
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
void RBM<InitializationRuleType, DataType, PolicyType>::Gradient(
    const arma::Mat<ElemType>& /*parameters*/,
    const size_t i,
    arma::Mat<ElemType>& gradient,
    const size_t batchSize)
{
  positiveGradient.zeros();
  negativeGradient.zeros();

  Phase(predictors.cols(i, i + batchSize - 1),
      positiveGradient);

  for (size_t i = 0; i < negSteps; ++i)
  {
    Gibbs(predictors.cols(i, i + batchSize - 1),
        negativeSamples);
    Phase(negativeSamples, tempNegativeGradient);

    negativeGradient += tempNegativeGradient;
  }

  gradient = ((negativeGradient / negSteps) - positiveGradient);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
void RBM<InitializationRuleType, DataType, PolicyType>::Shuffle()
{
  predictors = predictors.cols(arma::shuffle(arma::linspace<arma::uvec>(0,
      predictors.n_cols - 1, predictors.n_cols)));
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Archive>
void RBM<InitializationRuleType, DataType, PolicyType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(parameter);
  ar & BOOST_SERIALIZATION_NVP(visibleSize);
  ar & BOOST_SERIALIZATION_NVP(hiddenSize);
  ar & BOOST_SERIALIZATION_NVP(state);
  ar & BOOST_SERIALIZATION_NVP(numFunctions);
  ar & BOOST_SERIALIZATION_NVP(numSteps);
  ar & BOOST_SERIALIZATION_NVP(negSteps);
  ar & BOOST_SERIALIZATION_NVP(persistence);
  ar & BOOST_SERIALIZATION_NVP(poolSize);
  ar & BOOST_SERIALIZATION_NVP(visibleBias);
  ar & BOOST_SERIALIZATION_NVP(hiddenBias);
  ar & BOOST_SERIALIZATION_NVP(weight);
  ar & BOOST_SERIALIZATION_NVP(spikeBias);
  ar & BOOST_SERIALIZATION_NVP(slabPenalty);
  ar & BOOST_SERIALIZATION_NVP(radius);
  ar & BOOST_SERIALIZATION_NVP(visiblePenalty);

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    size_t shape = parameter.n_elem;
    positiveGradient.set_size(shape, 1);
    negativeGradient.set_size(shape, 1);
    negativeSamples.set_size(visibleSize, batchSize);
    tempNegativeGradient.set_size(shape, 1);
    spikeMean.set_size(hiddenSize, 1);
    spikeSamples.set_size(hiddenSize, 1);
    slabMean.set_size(poolSize, hiddenSize);
    positiveGradient.zeros();
    negativeGradient.zeros();
    tempNegativeGradient.zeros();
    reset = true;
  }
}

} // namespace ann
} // namespace mlpack
#endif
