/**
 * @file spike_slab_layer.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SPIKE_SLAB_HIDDEN_HPP
#define MLPACK_METHODS_ANN_LAYER_SPIKE_SLAB_HIDDEN_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::ann;

namespace mlpack{
namespace rbm{

template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class SpikeSlabLayer
{
 public:
  /* The visible layer of the rbm
   * network.
   *
   * @param: inSize num of visible neurons
   * @param: outSize num of hidden neurons
   * @param: poolSize num of pooling hidden neurons
   */
  SpikeSlabLayer(const size_t inSize, const size_t outSize,
      const size_t poolSize, const double radius, const bool typeVisible);

  // Reset the variables
  void Reset();

  /**
   * Sample the output given the input parameters
   * The sample are obtained from a distribution 
   * specified by the sampler.
   * P(h | v) --> visible.sample()
   * P(v | h) --> hidden.sample()
   *
   * @param input the input parameters for 
   * @param output samples from the parameters
   */
  void Sample(InputDataType&& input, OutputDataType&& output);

 /**
  * This function computes the P(v | s,h)
  * N(mean, variance)
  * mean = sum(w_i * s_i * h_i) * lambda^-1
  * variance = lambda^-1
  * We only samples that are in ball of size of R > max_t || v_t ||_2
  * t indexes over all training samples
  *
  * @param slab the slab variable(KxN)
  * @param hidden the spike variables(1xN)
  * @param output the sampled visible variable
  * @param returnMean return the mean of the distribution 
  */
  void Pvgivensh(arma::mat&& slab, arma::mat&& hidden, arma::mat&& outMean,
      arma::mat&& output);
  /**
  * This function computes the P(s | v, h)
  * \pi N(h_i * \alpha_i^-1 * w_i^T * v, \alpha_i^-1)
  * var = \sum_1^N(\alpha_i)
  * mu = var * \alpha_i * h_i * \alpha_i^-1 * w_i^T * v
  * which is equivalent to N (mu, var)
  *
  * @param visible the visible neuron(Dx1)
  * @param hidden the hidden layer (Nx1)
  * @param output the slab layer
  * @param returnMean return the mean of the distribution
  */
  void Psgivenvh(arma::mat&& visible, arma::mat&& hidden, arma::mat&& outMean,
      arma::mat&& output);

  //! Get the parameters.
  arma::mat const& Parameters() const { return weights; }
  //! Modify the parameters.
  arma::mat & Parameters() { return weights; }

  //! Get the weight variables
  arma::cube const& Weight() const { return weight; }
  arma::cube & Weight() { return weight; }

  //! Get the regulaliser associated with spike variables
  arma::mat const& SpikeBias() const { return spikeBias; }
  arma::mat & SpikeBias() { return spikeBias; }

  //! Get the regulaliser associated with slab variables
  arma::mat const& SlabBias() const { return slabBias; }
  arma::mat & SlabBias()  { return slabBias; }

  //! Get the regulaliser associated with visible variables
  arma::mat const& LambdaBias() const { return lambdaBias; }
  arma::mat & LambdaBias() { return lambdaBias; }

  //! Get the insize.
  size_t const& InSize() const { return inSize; }
  //! Get the outsize.
  size_t const& OutSize() const { return outSize; }
  //! Get the poolSize.
  size_t const& PoolSize() const { return poolSize; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored number of input units.
  const size_t inSize;
  //! Locally-stored number of visible units.
  size_t visibleSize;

  //! Locally-stored number of output units.
  const size_t outSize;

  //! Locally-stored number of hidden units.
  size_t hiddenSize;

  //! Locally-stored number of output units.
  const size_t poolSize;

  //! Locally stored radius for rejection sampling
  const double radius;
  //! Locally stored boolean variable indication type of layer
  const bool typeVisible;

  //! Locally-stored weight object.
  arma::mat weights;

  //! Locally-stored weight paramters.
  arma::cube weight;

  //! Locally-stored spike parameters.
  arma::mat spikeBias;

  //! Locally-store slab parameters
  arma::mat slabBias;

  //! Locally-store slab parameters
  arma::mat lambdaBias;

  //! Locally-stored state parmaeters storing the visible neurons
  arma::mat state;
  //! Locally-stored temp visible variable
  arma::mat visibleTemp;
  //! Locally-stored mean parameters for normal distribution
  arma::mat mean;
  //! Locally-stored variance parmaeters for normal distribution
  arma::mat variance;
  //! Locally-stored samples from the distribution
  arma::mat sample;
}; // class SpikeSlabHidden
} // namespace rbm
} // namespace mlpack
// Include implementation.
#include "spike_slab_layer_impl.hpp"
#endif
