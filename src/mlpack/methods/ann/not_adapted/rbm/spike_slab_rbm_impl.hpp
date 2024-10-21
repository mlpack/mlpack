/**
 * @file methods/ann/rbm/spike_slab_rbm_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_IMPL_HPP
#define MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_IMPL_HPP

#include "rbm.hpp"

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>


namespace mlpack {

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::Reset()
{
  size_t shape = (visibleSize * hiddenSize * poolSize) + visibleSize +
      hiddenSize;
  parameter.set_size(shape, 1);
  positiveGradient.set_size(shape, 1);
  negativeGradient.set_size(shape, 1);
  tempNegativeGradient.set_size(shape, 1);
  negativeSamples.set_size(visibleSize, batchSize);
  visibleMean.set_size(visibleSize, 1);
  spikeMean.set_size(hiddenSize, 1);
  spikeSamples.set_size(hiddenSize, 1);
  slabMean.set_size(poolSize, hiddenSize);

  // Weight shape D * K * N
  weight = arma::Cube<ElemType>(parameter.memptr(), visibleSize, poolSize,
      hiddenSize, false, false);
  // Spike bias shape N * 1
  spikeBias = DataType(parameter.memptr() + weight.n_elem, hiddenSize, 1,
      false, false);
  // Visible penalty 1 * 1 => D * D(when used)
  visiblePenalty = DataType(parameter.memptr() + weight.n_elem +
      spikeBias.n_elem, 1, 1, false, false);

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
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, double>
RBM<InitializationRuleType, DataType, PolicyType>::FreeEnergy(
    const arma::Mat<ElemType>& input)
{
  ElemType freeEnergy = 0.5 * visiblePenalty(0) * dot(input, input);

  freeEnergy -= 0.5 * hiddenSize * poolSize *
      std::log((2.0 * M_PI) / slabPenalty);

  for (size_t i = 0; i < hiddenSize; ++i)
  {
    ElemType sum = accu(square(input.t() * weight.slice(i))) /
        (2.0 * slabPenalty);
    freeEnergy -= SoftplusFunction::Fn(spikeBias(i) - sum);
  }

  return freeEnergy;
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::Phase(
    const InputType& input,
    DataType& gradient)
{
  arma::Cube<ElemType> weightGrad = arma::Cube<ElemType>
      (gradient.memptr(), visibleSize, poolSize, hiddenSize, false, false);

  DataType spikeBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  SpikeMean(input, spikeMean);
  SampleSpike(spikeMean, spikeSamples);
  SlabMean(input, spikeSamples, slabMean);

  for (size_t i = 0 ; i < hiddenSize; ++i)
  {
    weightGrad.slice(i) = input * repmat(slabMean.col(i).t(),
        input.n_cols, 1) * spikeMean(i);
  }

  spikeBiasGrad = spikeMean;
  // Setting visiblePenaltyGrad.
  gradient.row(weightGrad.n_elem + spikeBiasGrad.n_elem) = -0.5 * dot(
       input, input) / std::pow(input.n_cols, 2);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::SampleHidden(
    const arma::Mat<ElemType>& input,
    arma::Mat<ElemType>& output)
{
  output.set_size(hiddenSize + poolSize * hiddenSize, 1);

  DataType spike(output.memptr(), hiddenSize, 1, false, false);
  DataType slab(output.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  SpikeMean(input, spike);
  SampleSpike(spike, spike);
  SlabMean(input, spike, slab);
  SampleSlab(slab, slab);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::SampleVisible(
    arma::Mat<ElemType>& input,
    arma::Mat<ElemType>& output)
{
  const size_t numMaxTrials = 10;
  size_t k = 0;

  VisibleMean(input, visibleMean);
  output.set_size(visibleSize, 1);

  for (k = 0; k < numMaxTrials; ++k)
  {
    for (size_t i = 0; i < visibleSize; ++i)
    {
      output(i) = RandNormal(visibleMean(i), 1.0 / visiblePenalty(0));
    }
    if (norm(output, 2) < radius)
    {
      break;
    }
  }

  if (k == numMaxTrials)
  {
    Log::Warn << "Outputs are still not in visible unit "
        << norm(output, 2)
        << " terminating optimization."
        << std::endl;
  }
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::VisibleMean(
    InputType& input,
    DataType& output)
{
  output.zeros(visibleSize, 1);

  DataType spike(input.memptr(), hiddenSize, 1, false, false);
  DataType slab(input.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  for (size_t i = 0; i < hiddenSize; ++i)
  {
    output += weight.slice(i) * slab.col(i) * spike(i);
  }

  output = ((1.0 / visiblePenalty(0)) * output);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::HiddenMean(
    const InputType& input,
    DataType& output)
{
  output.set_size(hiddenSize + poolSize * hiddenSize, 1);

  DataType spike(output.memptr(), hiddenSize, 1, false, false);
  DataType slab(output.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  SpikeMean(input, spike);
  SampleSpike(spike, spikeSamples);
  SlabMean(input, spikeSamples, slab);
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::SpikeMean(
    const InputType& visible,
    DataType& spikeMean)
{
  for (size_t i = 0; i < hiddenSize; ++i)
  {
    spikeMean(i) = LogisticFunction::Fn(0.5 * (1.0 / slabPenalty) * accu(
        visible.t() * (weight.slice(i) * weight.slice(i).t()) * visible)
        / std::pow(visible.n_cols, 2) + spikeBias(i));
  }
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::SampleSpike(
    InputType& spikeMean,
    DataType& spike)
{
  for (size_t i = 0; i < hiddenSize; ++i)
  {
    spike(i) = RandBernoulli(spikeMean(i));
  }
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::SlabMean(
    const DataType& visible,
    DataType& spike,
    DataType& slabMean)
{
  for (size_t i = 0; i < hiddenSize; ++i)
  {
    slabMean.col(i) = arma::mean((1.0 / slabPenalty) * spike(i) *
        weight.slice(i).t() * visible, 1);
  }
}

template<
  typename InitializationRuleType,
  typename DataType,
  typename PolicyType
>
template<typename Policy, typename InputType>
std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
RBM<InitializationRuleType, DataType, PolicyType>::SampleSlab(
    InputType& slabMean,
    DataType& slab)
{
  for (size_t i = 0; i < hiddenSize; ++i)
  {
    for (size_t j = 0; j < poolSize; ++j)
    {
      slab(j, i) = RandNormal(slabMean(j, i), 1.0 / slabPenalty);
    }
  }
}

} // namespace mlpack

#endif
