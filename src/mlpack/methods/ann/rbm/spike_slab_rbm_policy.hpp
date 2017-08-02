/**
 * @file spike_slab_rbm_policy.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_HPP
#define MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

namespace mlpack {
namespace ann {

template<typename InputDataType = arma::mat, 
         typename OutputDataType = arma::mat>
class SpikeSlabRBMPolicy
{
 public:
  // Intialise the visible and hiddenl layer of the network
  SpikeSlabRBMPolicy(const size_t visibleSize,
      const size_t hiddenSize,
      const size_t poolSize,
      const double slabPenalty, 
      double radius);

  // Reset function
  void Reset();

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
  double FreeEnergy(InputDataType&& input);

  /**
   * Evaluate function is used by the Optimizer to 
   * find the perfomance of the network on the currentInput
   */
  double Evaluate(InputDataType& /*predictors*/, size_t /*i*/);

  /**
   * Positive Phase calculates the gradient on the
   * given data point. 
   *
   *
   * @param input the visible input
   * @param output the computed gradient
   */
  void PositivePhase(InputDataType&& input, OutputDataType&& gradient);

  /**
   * Negative Phase calculates the gradient on the
   * negative sample obtained for the given data point
   * using gibbs sampling. 
   *
   * @param input the visible input
   * @param output the computed gradient
   */
  void NegativePhase(InputDataType&& negativeSamples, OutputDataType&& gradient);

  /**
   * Visible Mean function calculates the mean of the
   * normal distribution of P(v| s,h).
   * Where the mean is given by \Lambda^{-1} \sum_{i=1}^N W_i * s_i * h_i 
   * 
   * @param input consists of spike and slab variables
   * @param output the mean of the of the Normal distribution
   */
  void VisibleMean(InputDataType&& input, OutputDataType&& output);

  /**
   * Hidden Mean function calculates the
   * normal distribution of P(s|v,h).
   * Where the mean is given by h_i*\alpha^{-1}*W_i^T*v 
   * variance is givenby \alpha^{-1}
   * 
   * @param input consists of visible input.
   * @param output consits of the spike samples and slab samples
   */
  void HiddenMean(InputDataType&& input, OutputDataType&& output);
  
  /**
   * Sample Visible function sample 
   * the visible outputs from the normal distribution with
   * mean \Lambda^{-1} \sum_{i=1}^N W_i * s_i * h_i and 
   * variance \Lambda^{-1}
   * 
   * @param input consists of spike and slab variables
   * @param output consits of visible layer.
   * @param norm norm used for rejection sampling
   */
  void SampleVisible(InputDataType&& input, OutputDataType&& output);

  /**
   * Sample Hidden function samples 
   * the slab outputs from the normal distribution with
   * mean by h_i*\alpha^{-1}*W_i^T*v  and 
   * variance \alpha&{-1}
   * 
   * @param input consists of visible and spike variables
   * @param output consits of slab units.
   */
  void SampleHidden(InputDataType&& input, OutputDataType&& output);

  // Serialize function
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Return the initial point for the optimization.
  const OutputDataType& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  OutputDataType& Parameters() { return parameter; }

  //! Get the weight variables
  arma::cube const& Weight() const { return weight; }
  arma::cube& Weight() { return weight; }

  //! Get the regulaliser associated with spike variables
  OutputDataType const& SpikeBias() const { return spikeBias; }
  OutputDataType& SpikeBias() { return spikeBias; }

  //! Get the regulaliser associated with slab variables
  double const& SlabPenalty() const { return slabPenalty; }

  //! Get the regulaliser associated with visible variables
  OutputDataType const& VisiblePenalty() const { return visiblePenalty; }
  OutputDataType& VisiblePenalty() { return visiblePenalty; }

  //! Get the visible size
  size_t const& VisibleSize() const { return visibleSize; }
  //! Get the hidden size
  size_t const& HiddenSize() const { return hiddenSize; }
  //! Get the pool size
  size_t const& PoolSize() const { return poolSize; }

 private:
  /**
   * Spike Mean function calculates the following distribution
   * P(h|v) which is given by
   *  sigm(v^T*W_i*\alpha_i^{-1}*W_i^T*v + b_i)
   *
   * @param visible the visible layer
   * @param spikeMean hidden layer
   */
  void SpikeMean(InputDataType&& visible, OutputDataType&& spikeMean);
  /**
   * Sample Spike function samples the spike
   * function using bernoulli distribution
   * @param spikeMean indicates P(h|v)
   * @param spike the sampled 0/1 spike variables
   */
  void SampleSpike(InputDataType&& spikeMean, OutputDataType&& spike);

  /**
   * SlabMean function calculates the mean of 
   * normal distribution of P(s|v,h).
   * Where the mean is given by h_i*\alpha^{-1}*W_i^T*v 
   * 
   * @param visible the visible layer units
   * @param spike the spike variables from hidden layer
   * @param slabMean the mean of the normal distribution
   */
  void SlabMean(InputDataType&& visible, InputDataType&& spike,
      OutputDataType&& slabMean);
  /**
   * SampleSlab function calculates the
   * normal distribution P(s|v,h).
   * Where the mean is given by h_i*\alpha^{-1}*W_i^T*v 
   * variance is givenby \alpha^{-1}
   *
   * @slabMean mean of the normal distribution
   * @slab sample slab variable from the normal distribution
   */
  void SampleSlab(InputDataType&& slabMean, OutputDataType&& slab);


 private:
  //! Locally stored parameters number of visible neurons
  size_t visibleSize;
  //! Locally stored parameters number of hidden neurons
  size_t hiddenSize;
  //! Locally stored parameters poolSize 
  size_t poolSize;
  //! Locally stored parameters
  OutputDataType parameter;
  //! Locally stored weight of the network (visibleSize * poolSize * hiddenSize)
  arma::cube weight;
  //! Locally stored spikeBias (hiddenSize * 1)
  InputDataType spikeBias;
  //! Locally stored slabPenalty
  double slabPenalty;
  //! Locally stored iverse of slabPenalty
  double invSlabPenalty;
  //! Locally stored radius used for rejection sampling
  double radius;
  //! Locally stored visible Penalty(1 * 1)
  InputDataType visiblePenalty;
  //! Locally stored saclar visible Penalty
  double scalarVisiblePenalty;
  //! Locally stored mean of the P(v | s,h)
  OutputDataType visibleMean;
  //! Locally stored mean of the P(v | h)
  OutputDataType spikeMean;
  //! Locally stored spike variables
  OutputDataType spikeSamples;
  //! Locally stored mean of the P(s | v, h)
  OutputDataType slabMean;
};
} // namespace ann
} // namespace mlpack

#include "spike_slab_rbm_policy_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_HPP
