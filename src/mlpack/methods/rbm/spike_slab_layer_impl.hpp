/**
 * @file spike_slab_hidden_impl.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RBM_SPIKE_SLAB_SPIKE_SLAB_LAYER_IMPL_HPP
#define MLPACK_METHODS_RBM_SPIKE_SLAB_SPIKE_SLAB_LAYER_IMPL_HPP
// In case it hasn't yet been included.
#include "spike_slab_layer.hpp"

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
using namespace mlpack::distribution;

namespace mlpack {
namespace rbm { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
SpikeSlabLayer<InputDataType, OutputDataType>::SpikeSlabLayer(
    const size_t inSize,
    const size_t outSize,
    const size_t poolSize,
    const bool typeVisible):
    inSize(inSize),
    outSize(outSize),
    poolSize(poolSize),
    typeVisible(typeVisible),
    radius(0)
{
  weights.set_size(
      inSize * ((outSize * inSize) + (poolSize * poolSize) + 1) + outSize,
      outSize * outSize);
}

template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Reset()
{
  if (typeVisible)
  {
    // Weight shape = k * d * n
    weight = arma::cube(weights.memptr(), poolSize, inSize, outSize, false,
        false);
    // slabBias shape = k * n ==> diagMat(slabBias.col(i)) = k * k
    slabBias = arma::mat(weights.memptr() + weight.n_elem,
        poolSize, outSize, false, false);
    
    // lambdaBias shape = d * 1 ==> diagMat(lambdaBias.col(0)) = d * d
    lambdaBias = arma::mat(weights.memptr() + poolSize * inSize * outSize +
        spikeBias.n_elem, inSize, 1, false, false);

    // spikeBias shape = 1 * N
    spikeBias = arma::mat(weights.memptr() + poolSize * inSize * outSize +
        spikeBias.n_elem + lambdaBias.n_elem, 1, outSize, false, false);
  }
  else
  {
   // Weight shape = k * d * n
    weight = arma::cube(weights.memptr(), poolSize, outSize, inSize, false,
        false);
    // slabBias shape = k * n ==> diagMat(slabBias.col(i)) = k * k
    slabBias = arma::mat(weights.memptr() + weight.n_elem,
        poolSize, inSize, false, false);
    
    // lambdaBias shape = d * 1 ==> diagMat(lambdaBias.col(0)) = d * d
    lambdaBias = arma::mat(weights.memptr() + poolSize * inSize * outSize +
        spikeBias.n_elem, outSize, 1, false, false);

    // spikeBias shape = 1 * N
    spikeBias = arma::mat(weights.memptr() + poolSize * inSize * outSize +
        spikeBias.n_elem + lambdaBias.n_elem, 1, inSize, false, false); 
  }
}

template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Sample(InputDataType&& input,
    OutputDataType&& output)
{
  if (!typeVisible)
  {
    if (state.is_empty())
      state = arma::zeros(outSize, 1);
    Psgivenvh(std::move(state), std::move(input), std::move(output));
    Pvgivensh(std::move(output), std::move(input), std::move(output));
  }
  else
  {
    for (size_t i = 0; i < outSize; i++)
    {
      if (output.is_empty())
        output = arma::zeros(outSize, 1);
      output.row(i) = LogisticFunction::Fn(arma::as_scalar(input.t() *
          weight.slice(i).t() * arma::diagmat(slabBias.col(i)).i() *
          weight.slice(i) * input + spikeBias.col(i)));
    }
  }
}
template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Pvgivensh(arma::mat&& slab,
    arma::mat&& hidden, arma::mat&& output)
{
  variance = arma::diagmat(lambdaBias).i();
  mean = arma::zeros(variance.n_rows, 1);
  for (size_t i = 0; i < inSize; i++)
  {
    mean += weight.slice(i).t() * slab.col(i) * arma::as_scalar(hidden(i));
  }
  mean = variance * mean;
  GaussianDistribution dist(mean, variance);
  sample = dist.Random();
  // Rejection sampling
  /*
  while( arma::norm(sample) > radius)
    sample = dist.Random();
  */

  output = sample;
  state = output;
}
template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Psgivenvh(
    arma::mat&& visible, arma::mat&& hidden, arma::mat&& output)
{
  output = arma::zeros(poolSize, inSize);
  for (size_t i = 0; i < inSize; i++)
  {
    variance = arma::diagmat(slabBias.col(i)).i();
    mean = arma::as_scalar(hidden(i)) * arma::diagmat(slabBias.col(i)).i()
        * weight.slice(i) * visible;
    GaussianDistribution dist(mean, variance);
    output.col(i) = dist.Random();
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SpikeSlabLayer<InputDataType, OutputDataType>::Serialize(
  Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(outSize, "outSize");
  ar & data::CreateNVP(poolSize, "poolSize");
}
} // namespace ann
} // namespace mlpack
#endif
