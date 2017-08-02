
#ifndef MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_IMPL_HPP
#define MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_IMPL_HPP

#include "spike_slab_rbm_policy.hpp"

namespace mlpack {
namespace ann {
template<typename InputDataType, typename OutputDataType>
inline SpikeSlabRBMPolicy<InputDataType, OutputDataType>
      ::SpikeSlabRBMPolicy(const size_t visibleSize,
      const size_t hiddenSize,
      const size_t poolSize,
      const double slabPenalty, 
      double radius):
      visibleSize(visibleSize),
      hiddenSize(hiddenSize),
      poolSize(poolSize),
      slabPenalty(slabPenalty),
      radius(radius)
{
  parameter.set_size(visibleSize * hiddenSize * poolSize +
      visibleSize + hiddenSize);

  visibleMean.set_size(visibleSize, 1);
  spikeMean.set_size(hiddenSize, 1);
  spikeSamples.set_size(hiddenSize, 1);
  slabMean.set_size(poolSize, hiddenSize);
  invSlabPenalty = 1.0 / slabPenalty;
};

// Reset function
template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>::Reset()
{
  // Weight shape D * K * N
  weight = arma::cube(parameter.memptr(), visibleSize, poolSize, hiddenSize,
      false, false);

  // spike bias shape N * 1
  spikeBias = arma::mat(parameter.memptr() + weight.n_elem, hiddenSize, 1,
      false, false);

  // visible penalty 1 * 1 => D * D(when used)
  visiblePenalty = arma::mat(parameter.memptr() + weight.n_elem +
      spikeBias.n_elem, 1, 1, false, false);
}

/**
 * Free energy of the spike and slab variable
 * the free energy of the ssRBM is given my
 * $v^t$$\Delta$v - $\sum_{i=1}^N$ 
 * $\log{ \sqrt{\frac{(-2\pi)^K}{\prod_{m=1}^{K}(\alpha_i)_m}}}$ -
 * $\sum_{i=1}^N \log(1+\exp( b_i +
 * \sum_{m=1}^k \frac{(v(w_i)_m^t)^2}{2(\alpha_i)_m})$
 *
 * @param input the visible layer
 */ 
template<typename InputDataType, typename OutputDataType>
inline double SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::FreeEnergy(InputDataType&& input)
{
  assert(input.n_rows == visibleSize);
  assert(input.n_cols == 1);

  scalarVisiblePenalty = visiblePenalty(0, 0);

  double freeEnergy = 0.5 * arma::as_scalar(scalarVisiblePenalty * input.t() *
      input);

  freeEnergy -= 0.5 * hiddenSize * poolSize *
      std::log((2.0 * M_PI) / slabPenalty);

  for (size_t i = 0; i < hiddenSize; i++)
  {
    double sum = 0;

    for (size_t k = 0; k < poolSize; k++)
    {
      sum += arma::as_scalar(input.t() * weight.slice(i).col(k)) *
          arma::as_scalar(input.t() * weight.slice(i).col(k)) /
          (2.0 * slabPenalty);
    }

    freeEnergy -= SoftplusFunction::Fn(spikeBias(i) - sum);
  }

  return freeEnergy;
}

template<typename InputDataType, typename OutputDataType>
inline double SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::Evaluate(InputDataType& /*predictors*/,
    size_t /*i*/)
{
  // Return 0 here since we don't have evaluate in case of persistence
  return 0;
}

/**
 * Gradient function calculates the gradient for the spike and
 * slab RBM.
 *
 * @param input the visible input
 * @param output the computed gradient
 */
template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::PositivePhase(InputDataType&& input,
    OutputDataType&& gradient)
{
  assert(input.n_rows == visibleSize);
  assert(input.n_cols == 1);
  arma::cube weightGrad = arma::cube(gradient.memptr(), visibleSize, poolSize,
      hiddenSize, false, false);

  arma::mat spikeBiasGrad = arma::mat(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  arma::mat visiblePenaltyGrad = arma::mat(gradient.memptr() +
      weightGrad.n_elem + spikeBiasGrad.n_elem, 1, 1, false, false);

  SpikeMean(std::move(input), std::move(spikeMean));
  SampleSpike(std::move(spikeMean), std::move(spikeSamples));
  SlabMean(std::move(input), std::move(spikeSamples), std::move(slabMean));

  // positive weight gradient
  for (size_t i = 0 ; i < hiddenSize; i++)
    weightGrad.slice(i) = input * slabMean.col(i).t() * spikeMean(i);

  // positive hidden bias gradient
  for (size_t i = 0; i < hiddenSize; i++)
    spikeBiasGrad(i) = spikeMean(i);
  // positive lambda bias
  visiblePenaltyGrad = -0.5 * input.t() * input;
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::NegativePhase(InputDataType&& negativeSamples,
    OutputDataType&& gradient)
{
  assert(negativeSamples.n_rows == visibleSize);
  assert(negativeSamples.n_cols == 1);
  arma::cube weightGrad = arma::cube(gradient.memptr(), visibleSize, poolSize,
      hiddenSize, false, false);

  arma::mat spikeBiasGrad = arma::mat(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  arma::mat visiblePenaltyGrad = arma::mat(gradient.memptr() +
      weightGrad.n_elem + spikeBiasGrad.n_elem, 1, 1, false, false);

  SpikeMean(std::move(negativeSamples), std::move(spikeMean));
  SampleSpike(std::move(spikeMean), std::move(spikeSamples));
  SlabMean(std::move(negativeSamples), std::move(spikeSamples),
      std::move(slabMean));

  // positive weight gradient
  for (size_t i = 0 ; i < hiddenSize; i++)
    weightGrad.slice(i) = negativeSamples * slabMean.col(i).t() * spikeMean(i);

  // positive hidden bias gradient
  for (size_t i = 0; i < hiddenSize; i++)
    spikeBiasGrad(i) = spikeMean(i);
  // positive lambda bias
  visiblePenaltyGrad = -0.5 * negativeSamples.t() * negativeSamples;
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::SpikeMean(InputDataType&& visible,
    OutputDataType&& spikeMean)
{
  assert(visible.n_rows == visibleSize);
  assert(visible.n_cols == 1);

  assert(spikeMean.n_rows == hiddenSize);
  assert(spikeMean.n_cols == 1);

  for (size_t i = 0; i < hiddenSize; i++)
  {
    spikeMean(i) = LogisticFunction::Fn(0.5 * invSlabPenalty *
        arma::as_scalar(visible.t() * weight.slice(i) * weight.slice(i).t() *
        visible) + spikeBias(i));
  }
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::SampleSpike(InputDataType&& spikeMean,
    OutputDataType&& spike)
{
  assert(spikeMean.n_rows == hiddenSize);
  assert(spikeMean.n_cols == 1);

  assert(spike.n_rows == hiddenSize);
  assert(spike.n_cols == 1);

  for (size_t i = 0; i < hiddenSize; i++)
    spike(i) = math::RandBernoulli(spikeMean(i));
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::SlabMean(InputDataType&& visible,
    InputDataType&& spike,
    OutputDataType&& slabMean)
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
    slabMean.col(i) = invSlabPenalty * spike(i) *
        weight.slice(i).t() * visible;
  }
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::SampleSlab(InputDataType&& slabMean,
    OutputDataType&& slab)
{
  assert(slabMean.n_rows == poolSize);
  assert(slabMean.n_cols == hiddenSize);

  assert(slab.n_rows == poolSize);
  assert(slab.n_cols == hiddenSize);

  for (size_t i = 0; i < hiddenSize; i++)
  {
    for (size_t j = 0; j < poolSize; j++)
    {
      slab(j, i) = math::RandNormal(slabMean(j, i), invSlabPenalty);
    }
  }
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::VisibleMean(InputDataType&& input,
    OutputDataType&& output)
{
  assert(input.n_elem == hiddenSize + poolSize * hiddenSize);
  scalarVisiblePenalty = visiblePenalty(0, 0);
  output.set_size(visibleSize, 1);
  output.zeros();

  arma::mat spike(input.memptr(), hiddenSize, 1, false, false);
  arma::mat slab(input.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  for (size_t i = 0; i < hiddenSize; i++)
    output += weight.slice(i) * slab.col(i) * spike(i);

  output = ((1.0 / scalarVisiblePenalty) * output);
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::HiddenMean(InputDataType&& input,
    OutputDataType&& output)
{
  assert(input.n_elem == visibleSize);
  output.set_size(hiddenSize + poolSize * hiddenSize, 1);

  arma::mat spike(output.memptr(), hiddenSize, 1, false, false);
  arma::mat slab(output.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  SpikeMean(std::move(input), std::move(spike));
  SampleSpike(std::move(spike), std::move(spikeSamples));
  SlabMean(std::move(input), std::move(spikeSamples), std::move(slab));
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::SampleVisible(InputDataType&& input,
    OutputDataType&& output)
{
  const size_t numMaxTrials = 10;
  scalarVisiblePenalty = visiblePenalty(0,0);

  VisibleMean(std::move(input), std::move(visibleMean));

  output.set_size(visibleSize, 1);

  for (size_t k = 0; k < numMaxTrials; k++)
  {
    for (size_t i = 0; i < visibleSize; i++)
    {
      assert(scalarVisiblePenalty > 0);
      output(i) = math::RandNormal(visibleMean(i), 1.0 / scalarVisiblePenalty);
    }
    if (arma::norm(output, 2) < radius)
      break;
  }
}

template<typename InputDataType, typename OutputDataType>
inline void SpikeSlabRBMPolicy<InputDataType, OutputDataType>
    ::SampleHidden(InputDataType&& input,
    OutputDataType&& output)
{
  assert(input.n_elem == visibleSize);
  output.set_size(hiddenSize + poolSize * hiddenSize, 1);

  arma::mat spike(output.memptr(), hiddenSize, 1, false, false);
  arma::mat slab(output.memptr() + hiddenSize, poolSize, hiddenSize, false,
      false);

  SpikeMean(std::move(input), std::move(spike));
  SampleSpike(std::move(spike), std::move(spike));
  SlabMean(std::move(input), std::move(spike), std::move(slab));
  SampleSlab(std::move(slab), std::move(slab));
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SpikeSlabRBMPolicy<InputDataType, OutputDataType>::Serialize(Archive& ar,
    const unsigned int /* version */)
{
  ar & data::CreateNVP(visibleSize, "visibleSize");
  ar & data::CreateNVP(hiddenSize, "hiddenSize");
  ar & data::CreateNVP(poolSize, "poolSize");
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(weight, "weight");
  ar & data::CreateNVP(spikeBias, "spikeBias");
  ar & data::CreateNVP(slabPenalty, "slabPenalty");
  ar & data::CreateNVP(radius, "radius");
  ar & data::CreateNVP(visiblePenalty, "visiblePenalty");

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
