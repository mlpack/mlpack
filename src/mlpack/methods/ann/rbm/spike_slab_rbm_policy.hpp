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

namespace mlpack {
namespace ann {

class SpikeSlabRBMPolicy
{
 public:
  // Intialise the visible and hiddenl layer of the network
  SpikeSlabRBMPolicy(const size_t visibleSize,
      const size_t hiddenSize,
      const size_t poolSize,
      const arma::mat& slabPenalty, 
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
  double FreeEnergy(arma::mat&& input);

  double Evaluate(arma::mat& /*predictors*/, size_t /*i*/);

  /**
   * Gradient function calculates the gradient for the spike and
   * slab RBM.
   *
   * @param input the visible input
   * @param output the computed gradient
   */
  void PositivePhase(arma::mat&& input, arma::mat&& gradient);

  void NegativePhase(arma::mat&& negativeSamples, arma::mat&& gradient);

  void VisibleMean(arma::mat&& input, arma::mat&& output);

  void HiddenMean(arma::mat&& input, arma::mat&& output);
  
  void SampleVisible(arma::mat&& input, arma::mat&& output);

  void SampleHidden(arma::mat&& input, arma::mat&& output);

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  //! Get the weight variables
  arma::cube const& Weight() const { return weight; }
  arma::cube& Weight() { return weight; }

  //! Get the regulaliser associated with spike variables
  arma::mat const& SpikeBias() const { return spikeBias; }
  arma::mat& SpikeBias() { return spikeBias; }

  //! Get the regulaliser associated with slab variables
  arma::mat const& SlabPenalty() const { return slabPenalty; }

  //! Get the regulaliser associated with visible variables
  arma::mat const& VisiblePenalty() const { return visiblePenalty; }
  arma::mat& VisiblePenalty() { return visiblePenalty; }

 private:
  void SpikeMean(arma::mat&& visible, arma::mat&& spikeMean);
  void SampleSpike(arma::mat&& spikeMean, arma::mat&& spike);

  void SlabMean(arma::mat&& visible, arma::mat&& spike, arma::mat&& slabMean);
  void SampleSlab(arma::mat&& slabMean, arma::mat&& slab);


 private:
  //! Locally stored parameters number of visible neurons
  size_t visibleSize;
  //! Locally stored parameters number of hidden neurons
  size_t hiddenSize;

  size_t poolSize;

  //! Locally stored parameters
  arma::mat parameter;

  arma::cube weight;

  arma::mat spikeBias;

  arma::mat slabPenalty;

  double radius;

  arma::mat visiblePenalty;

  arma::mat spikeMean;

  arma::mat spikeSamples;

  arma::mat slabMean;
};
} // namespace ann
} // namespace mlpack

#include "spike_slab_rbm_policy_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_SPIKE_SLAB_RBM_POLICY_HPP
