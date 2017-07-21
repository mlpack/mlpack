/**
 * @file spike_slab_hidden_impl.hpp
 * @author Kris Singh
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
    const double radius,
    const bool typeVisible):
    inSize(inSize),
    outSize(outSize),
    poolSize(poolSize),
    radius(radius),
    typeVisible(typeVisible)
{}

template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Reset()
{
  if (typeVisible)
  {
    visibleSize = inSize;
    hiddenSize = outSize;
  }
  else
  {
    visibleSize = outSize;
    hiddenSize = inSize;
  }
  // Weight shape = k * d * n
  weight = arma::cube(weights.memptr(), poolSize,
      visibleSize, hiddenSize, false,
      false);
  // slabBias shape = k * n ==> diagMat(slabBias.col(i)) = k * k
  slabBias = arma::mat(weights.memptr() + weight.n_elem,
      poolSize, hiddenSize, false, false);

  // lambdaBias shape = d * 1 ==> diagMat(lambdaBias.col(0)) = d * d
  lambdaBias = arma::mat(weights.memptr() + weight.n_elem + slabBias.n_elem,
      visibleSize, 1, false, false);

  // spikeBias shape = 1 * N
  spikeBias = arma::mat(weights.memptr() + weight.n_elem + slabBias.n_elem +
      lambdaBias.n_elem, 1, hiddenSize, false, false);
}

template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Sample(InputDataType&&
    input, OutputDataType&& output)
{
  if (!typeVisible)
  {
    if (state.is_empty())
      state = arma::zeros(visibleSize, 1);
    Psgivenvh(std::move(state), std::move(input), std::move(mean),
        std::move(output));
    Pvgivensh(std::move(output), std::move(input), std::move(mean),
        std::move(output));
  }
  else
  {
    for (size_t i = 0; i < hiddenSize; i++)
    {
      if (output.is_empty())
        output = arma::zeros(hiddenSize, 1);
      output(i) = LogisticFunction::Fn(0.5 * arma::as_scalar(input.t() *
          weight.slice(i).t() * arma::diagmat(slabBias.col(i)).i() *
          weight.slice(i) * input + spikeBias.col(i)));
      output(i) = math::RandBernoulli(output(i));
    }
  }
}
template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Pvgivensh(arma::mat&& slab,
    arma::mat&& hidden, arma::mat&& outMean, arma::mat&& output)
{
  variance = arma::diagmat(lambdaBias.col(0)).i();
  mean = arma::zeros(variance.n_rows, 1);
  sample.set_size(variance.n_rows, 1);
  for (size_t i = 0; i < weight.n_slices; i++)
  {
    mean += weight.slice(i).t() * slab.col(i) * arma::as_scalar(hidden(i));
  }
  mean = variance * mean;
  for (size_t i = 0; i < mean.n_rows; i++)
      sample(i) = math::RandNormal(mean(i), variance(i, i));
  // Rejection sampling
  for (size_t i = 0; i < hiddenSize; i++)
    if (arma::norm(sample) > radius)
      for (size_t i = 0; i < mean.n_rows; i++)
        sample(i) = math::RandNormal(mean(i), variance(i, i));
  output = sample;
  outMean = mean;
  state = output;
}
template<typename InputDataType, typename OutputDataType>
void SpikeSlabLayer<InputDataType, OutputDataType>::Psgivenvh(
    arma::mat&& visible, arma::mat&& hidden, arma::mat&& outMean,
    arma::mat&& output)
{
  output = arma::zeros(slabBias.n_rows, slabBias.n_cols);
  for (size_t i = 0; i < slabBias.n_cols; i++)
  {
    variance = arma::diagmat(slabBias.col(i)).i();
    mean = arma::as_scalar(hidden(i)) * variance
        * weight.slice(i) * visible;
    for (size_t j = 0; j < slabBias.n_rows; j++)
    {
      output(j, i) = math::RandNormal(mean(j), variance(j, j));
    }
  }
  outMean = mean;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SpikeSlabLayer<InputDataType, OutputDataType>::Serialize(
  Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(visibleSize, "inSize");
  ar & data::CreateNVP(hiddenSize, "outSize");
  ar & data::CreateNVP(poolSize, "poolSize");
}
} // namespace rbm
} // namespace mlpack
#endif
