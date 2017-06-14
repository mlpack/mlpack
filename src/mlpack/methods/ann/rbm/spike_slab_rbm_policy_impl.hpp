
#ifndef MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_IMPL_HPP
#define MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_IMPL_HPP

#include "spike_slab_rbm_policy.hpp"

namespace mlpack {
namespace ann {
template<typename DataType>
SpikeSlabRBMPolicy<DataType>::SpikeSlabRBMPolicy(
    const size_t visibleSize,
    const size_t hiddenSize,
    const size_t poolSize,
    const ElemType slabPenalty,
    ElemType radius):
    visibleSize(visibleSize),
    hiddenSize(hiddenSize),
    poolSize(poolSize),
    slabPenalty(slabPenalty),
    radius(2 * radius)
{
  parameter.set_size(visibleSize * hiddenSize * poolSize +
      visibleSize + hiddenSize);

  visibleMean.set_size(visibleSize, 1);
  spikeMean.set_size(hiddenSize, 1);
  spikeSamples.set_size(hiddenSize, 1);
  slabMean.set_size(poolSize, hiddenSize);
};

// Reset function
template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::Reset()
{
  // Weight shape D * K * N
  weight = arma::Cube<ElemType>(parameter.memptr(),
      visibleSize, poolSize, hiddenSize,
      false, false);

  // spike bias shape N * 1
  spikeBias = DataType(parameter.memptr() + weight.n_elem, hiddenSize, 1,
      false, false);

  // visible penalty 1 * 1 => D * D(when used)
  visiblePenalty = DataType(parameter.memptr() + weight.n_elem +
      spikeBias.n_elem, 1, 1, false, false);
}

template<typename DataType>
typename SpikeSlabRBMPolicy<DataType>::ElemType
SpikeSlabRBMPolicy<DataType>::FreeEnergy(DataType&& input)
{
  assert(input.n_rows == visibleSize);
  assert(input.n_cols == 1);


  ElemType freeEnergy = 0.5 * arma::as_scalar(visiblePenalty(0) * input.t() *
      input);

  freeEnergy -= 0.5 * hiddenSize * poolSize *
      std::log((2.0 * M_PI) / slabPenalty);

  for (size_t i = 0; i < hiddenSize; i++)
  {
    ElemType sum = 0;
    sum = arma::accu(arma::square(input.t() * weight.slice(i))) /
        (2.0 * slabPenalty);
    freeEnergy -= SoftplusFunction::Fn(spikeBias(i) - sum);
  }

  return freeEnergy;
}

template<typename DataType>
typename SpikeSlabRBMPolicy<DataType>::ElemType
SpikeSlabRBMPolicy<DataType>::Evaluate(DataType& /*predictors*/,
                                       size_t /*i*/)
{
  // Return 0 here since we don't have evaluate in case of persistence
  return 0;
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::PositivePhase(
    DataType&& input,
    DataType&& gradient)
{
  assert(input.n_rows == visibleSize);
  assert(input.n_cols == 1);
  arma::Cube<ElemType> weightGrad = arma::Cube<ElemType>
      (gradient.memptr(), visibleSize, poolSize, hiddenSize, false, false);

  DataType spikeBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  DataType visiblePenaltyGrad = DataType(gradient.memptr() +
      weightGrad.n_elem + spikeBiasGrad.n_elem, 1, 1, false, false);

  SpikeMean(std::move(input), std::move(spikeMean));
  SampleSpike(std::move(spikeMean), std::move(spikeSamples));
  SlabMean(std::move(input), std::move(spikeSamples), std::move(slabMean));

  for (size_t i = 0 ; i < hiddenSize; i++)
    weightGrad.slice(i) = input * slabMean.col(i).t() * spikeMean(i);

  spikeBiasGrad = spikeMean;

  visiblePenaltyGrad = -0.5 * input.t() * input;
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::NegativePhase(
    DataType&& negativeSamples,
    DataType&& gradient)
{
  assert(negativeSamples.n_rows == visibleSize);
  assert(negativeSamples.n_cols == 1);
  arma::Cube<ElemType> weightGrad = arma::Cube<ElemType>
      (gradient.memptr(), visibleSize, poolSize, hiddenSize, false, false);

  DataType spikeBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  DataType visiblePenaltyGrad = DataType(gradient.memptr() +
      weightGrad.n_elem + spikeBiasGrad.n_elem, 1, 1, false, false);

  SpikeMean(std::move(negativeSamples), std::move(spikeMean));
  SampleSpike(std::move(spikeMean), std::move(spikeSamples));
  SlabMean(std::move(negativeSamples), std::move(spikeSamples),
      std::move(slabMean));

  for (size_t i = 0 ; i < hiddenSize; i++)
    weightGrad.slice(i) = negativeSamples * slabMean.col(i).t() * spikeMean(i);

  spikeBiasGrad = spikeMean;

  visiblePenaltyGrad = -0.5 * negativeSamples.t() * negativeSamples;
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::SpikeMean(
    DataType&& visible,
    DataType&& spikeMean)
{
  assert(visible.n_rows == visibleSize);
  assert(visible.n_cols == 1);

  assert(spikeMean.n_rows == hiddenSize);
  assert(spikeMean.n_cols == 1);

  for (size_t i = 0; i < hiddenSize; i++)
  {
    spikeMean(i) = LogisticFunction::Fn(0.5 * (1.0 / slabPenalty) *
        arma::as_scalar(visible.t() * weight.slice(i) * weight.slice(i).t() *
        visible) + spikeBias(i));
  }
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::SampleSpike(
    DataType&& spikeMean,
    DataType&& spike)
{
  assert(spikeMean.n_rows == hiddenSize);
  assert(spikeMean.n_cols == 1);

  assert(spike.n_rows == hiddenSize);
  assert(spike.n_cols == 1);

  for (size_t i = 0; i < hiddenSize; i++)
    spike(i) = math::RandBernoulli(spikeMean(i));
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::SlabMean(
    DataType&& visible,
    DataType&& spike,
    DataType&& slabMean)
{
  assert(visible.n_rows == visibleSize);
  assert(visible.n_cols == 1);

  assert(spike.n_rows == hiddenSize);
  assert(spike.n_cols == 1);

  assert(slabMean.n_rows == poolSize);
  assert(slabMean.n_cols == hiddenSize);

  assert(weight.n_rows == visibleSize);
  assert(weight.n_cols == poolSize);

  for (size_t i = 0; i < hiddenSize; i++)
  {
    slabMean.col(i) = (1.0 / slabPenalty) * spike(i) *
        weight.slice(i).t() * visible;
  }
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::SampleSlab(
    DataType&& slabMean,
    DataType&& slab)
{
  assert(slabMean.n_rows == poolSize);
  assert(slabMean.n_cols == hiddenSize);

  assert(slab.n_rows == poolSize);
  assert(slab.n_cols == hiddenSize);

  for (size_t i = 0; i < hiddenSize; i++)
  {
    for (size_t j = 0; j < poolSize; j++)
    {
      slab(j, i) = math::RandNormal(slabMean(j, i), 1.0 / slabPenalty);
    }
  }
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::VisibleMean(
    DataType&& input,
    DataType&& output)
{
  assert(input.n_elem == hiddenSize + poolSize * hiddenSize);
  output.zeros(visibleSize, 1);

  DataType spike(input.memptr(), hiddenSize, 1, false, false);
  DataType slab(input.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  for (size_t i = 0; i < hiddenSize; i++)
    output += weight.slice(i) * slab.col(i) * spike(i);

  output = ((1.0 / visiblePenalty(0)) * output);
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::HiddenMean(
    DataType&& input,
    DataType&& output)
{
  assert(input.n_elem == visibleSize);
  output.set_size(hiddenSize + poolSize * hiddenSize, 1);

  DataType spike(output.memptr(), hiddenSize, 1, false, false);
  DataType slab(output.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  SpikeMean(std::move(input), std::move(spike));
  SampleSpike(std::move(spike), std::move(spikeSamples));
  SlabMean(std::move(input), std::move(spikeSamples), std::move(slab));
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::SampleVisible(
    DataType&& input,
    DataType&& output)
{
  const size_t numMaxTrials = 10;
  size_t k = 0;

  VisibleMean(std::move(input), std::move(visibleMean));

  output.set_size(visibleSize, 1);

  assert(visiblePenalty(0) > 0);

  for (k = 0; k < numMaxTrials; k++)
  {
    for (size_t i = 0; i < visibleSize; i++)
    {
      output(i) = math::RandNormal(visibleMean(i), 1.0 / visiblePenalty(0));
    }
    if (arma::norm(output, 2) < radius)
      break;
  }

  if (k == numMaxTrials)
  {
    Log::Warn << "Outputs are still not in visible unit "
        << arma::norm(output, 2)
        << " terminating optimization."
        << std::endl;
    return;
  }
}

template<typename DataType>
void SpikeSlabRBMPolicy<DataType>::SampleHidden(
    DataType&& input,
    DataType&& output)
{
  assert(input.n_elem == visibleSize);
  output.set_size(hiddenSize + poolSize * hiddenSize, 1);

  DataType spike(output.memptr(), hiddenSize, 1, false, false);
  DataType slab(output.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  SpikeMean(std::move(input), std::move(spike));
  SampleSpike(std::move(spike), std::move(spike));
  SlabMean(std::move(input), std::move(spike), std::move(slab));
  SampleSlab(std::move(slab), std::move(slab));
}

template<typename DataType>
template<typename Archive>
void SpikeSlabRBMPolicy<DataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(visibleSize);
  ar & BOOST_SERIALIZATION_NVP(hiddenSize);
  ar & BOOST_SERIALIZATION_NVP(poolSize);
  ar & BOOST_SERIALIZATION_NVP(parameter);
  ar & BOOST_SERIALIZATION_NVP(weight);
  ar & BOOST_SERIALIZATION_NVP(spikeBias);
  ar & BOOST_SERIALIZATION_NVP(slabPenalty);
  ar & BOOST_SERIALIZATION_NVP(radius);
  ar & BOOST_SERIALIZATION_NVP(visiblePenalty);

  if (Archive::is_loading::value)
  {
    spikeMean.set_size(hiddenSize, 1);
    spikeSamples.set_size(hiddenSize, 1);
    slabMean.set_size(poolSize, hiddenSize);
    Reset();
  }
}

} // namespace ann
} // namespace mlpack


#endif // MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_IMPL_HPP
