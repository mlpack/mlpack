/**
 * @file spike_slab_rbm_policy.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_HPP
#define MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

namespace mlpack {
namespace ann {
/**
 * The SpikeSlabRBMPolicy class.
 *
 * @tparam DataType the type of matrix to be used.
 */
template<typename DataType = arma::mat>
class SpikeSlabRBMPolicy
{
 public:
  typedef typename DataType::elem_type ElemType;
  /**
   * Intialise the visible and hidden layer of the network
   *
   * @param visibleSize Number of visible neurons.
   * @param hiddenSize Number of hidden neurons.
   * @param poolSize Number of hidden neurons to pool together.
   * @param slabPenalty Regulariser of slab varaibles.
   * @param radius Feasible regions for visible layer samples.
   */
  SpikeSlabRBMPolicy(const size_t visibleSize,
      const size_t hiddenSize,
      const size_t poolSize,
      const ElemType slabPenalty,
      ElemType radius);

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
   * @param input The visible layer neurons.
   */ 
  ElemType FreeEnergy(DataType&& input);

  /**
   * Evaluate function is used by the Optimizer to 
   * find the perfomance of the network on the currentInput
   *
   * @param predictors The training data used.
   * @param i The idx of the current input.
   */
  ElemType Evaluate(DataType& /*predictors*/, size_t /*i*/);

  /**
   * Calculate the Gradient of the RBM network on the 
   * visible input from the training data.
   * 
   * @param input The visible layer neurons.
   * @param gradient Stores the gradient of the rbm network.
   */
  void PositivePhase(DataType&& input, DataType&& gradient);

  /**
   * Calculate the Gradient of the RBM network on the sampled
   * visible input from gibbs sampling
   *
   * @param input The visible layer neurons.
   * @param output Stores the computed gradient of the rbm network.
   */
  void NegativePhase(DataType&& negativeSamples, DataType&& gradient);

  /**
   * Visible Mean function calculates the mean of the
   * normal distribution of P(v| s,h).
   * Where the mean is given by \Lambda^{-1} \sum_{i=1}^N W_i * s_i * h_i 
   * 
   * @param input Consists of both the spike and slab variables.
   * @param output Mean of the of the Normal distribution.
   */
  void VisibleMean(DataType&& input, DataType&& output);

  /**
   * Hidden Mean function calculates the mean of the 
   * normal distribution of P(s|v,h).
   * Where the mean is given by h_i*\alpha^{-1}*W_i^T*v 
   * variance is givenby \alpha^{-1}
   * 
   * @param input Visible layer neurons.
   * @param output Consits of both the spike samples and slab samples.
   */
  void HiddenMean(DataType&& input, DataType&& output);

  /**
   * Sample Visible function samples
   * the visible layer from the normal distribution with
   * mean \Lambda^{-1} \sum_{i=1}^N W_i * s_i * h_i and 
   * variance \Lambda^{-1}
   * 
   * @param input Consists of spike and slab variables.
   * @param output Sampled visible layer neurons.
   */
  void SampleVisible(DataType&& input, DataType&& output);

  /**
   * Sample Hidden function samples 
   * the slab outputs from the normal distribution with
   * mean by h_i*\alpha^{-1}*W_i^T*v  and 
   * variance \alpha&{-1}
   * 
   * @param input Consists of both visible and spike variables.
   * @param output Smapled slab neurons.
   */
  void SampleHidden(DataType&& input, DataType&& output);

  //! Serialize function.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  //! Return the parameters of the network.
  const DataType& Parameters() const { return parameter; }
  //! Modify the parameters of the network.
  DataType& Parameters() { return parameter; }

  //! Get the weight of the network.
  arma::cube const& Weight() const { return weight; }
  //! Modify the weights of the network.
  arma::cube& Weight() { return weight; }

  //! Get the regulaliser associated with spike variables.
  DataType const& SpikeBias() const { return spikeBias; }
  //! Modify the regulaliser associated with spike variables.
  DataType& SpikeBias() { return spikeBias; }

  //! Get the regulaliser associated with slab variables.
  ElemType const& SlabPenalty() const { return 1.0 / slabPenalty; }

  //! Get the regulaliser associated with visible variables.
  DataType const& VisiblePenalty() const { return visiblePenalty; }
  //! Modify the regulaliser associated with visible variables.
  DataType& VisiblePenalty() { return visiblePenalty; }

  //! Get the visible size.
  size_t const& VisibleSize() const { return visibleSize; }
  //! Get the hidden size.
  size_t const& HiddenSize() const { return hiddenSize; }
  //! Get the pool size.
  size_t const& PoolSize() const { return poolSize; }

 private:
  /**
   * Spike Mean function calculates the mean of the following
   * distribution P(h|v) where mean  is given by
   * sigm(v^T*W_i*\alpha_i^{-1}*W_i^T*v + b_i)
   *
   * @param visible The visible layer neurons.
   * @param spikeMean Indicates P(h|v).
   */
  void SpikeMean(DataType&& visible, DataType&& spikeMean);
  /**
   * Sample Spike function samples the spike
   * function using bernoulli distribution
   * @param spikeMean Indicates P(h|v).
   * @param spike Sampled binary spike variables.
   */
  void SampleSpike(DataType&& spikeMean, DataType&& spike);

  /**
   * SlabMean function calculates the mean of 
   * normal distribution of P(s|v,h).
   * Where the mean is given by h_i*\alpha^{-1}*W_i^T*v 
   * 
   * @param visible The visible layer neurons.
   * @param spike The spike variables from hidden layer.
   * @param slabMean The mean of the normal distribution of slab neurons.
   */
  void SlabMean(DataType&& visible, DataType&& spike, DataType&& slabMean);
  /**
   * SampleSlab function samples from the
   * normal distribution P(s|v,h).
   * Where the mean is given by h_i*\alpha^{-1}*W_i^T*v 
   * variance is givenby \alpha^{-1}
   *
   * @slabMean Mean of the normal distribution of the slab neurons.
   * @slab Sampled slab variable from the normal distribution.
   */
  void SampleSlab(DataType&& slabMean, DataType&& slab);

  //! Locally stored number of visible neurons.
  size_t visibleSize;
  //! Locally stored number of hidden neurons.
  size_t hiddenSize;
  //! Locally stored variable poolSize.
  size_t poolSize;
  //! Locally stored parameters.
  DataType parameter;
  //! Locally stored weight of the network(visibleSize * poolSize * hiddenSize).
  arma::Cube<ElemType> weight;
  //! Locally stored spikeBias (hiddenSize * 1).
  DataType spikeBias;
  //! Locally stored slabPenalty.
  ElemType slabPenalty;
  //! Locally stored radius used for rejection sampling.
  ElemType radius;
  //! Locally stored visible Penalty(1 * 1).
  DataType visiblePenalty;
  //! Locally stored mean of the P(v | s,h).
  DataType visibleMean;
  //! Locally stored mean of the P(v | h).
  DataType spikeMean;
  //! Locally stored spike variables.
  DataType spikeSamples;
  //! Locally stored mean of the P(s | v, h).
  DataType slabMean;
};
} // namespace ann
} // namespace mlpack

#include "spike_slab_rbm_policy_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_HPP
