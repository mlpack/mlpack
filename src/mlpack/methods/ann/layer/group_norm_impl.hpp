/**
 * @file methods/ann/layer/group_norm_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the Group Normalization class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_GROUPNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GROUPNORM_IMPL_HPP

// In case it is not included.
#include "group_norm.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */


template<typename InputDataType, typename OutputDataType>
GroupNorm<InputDataType, OutputDataType>::GroupNorm() :
    inRowSize(0),
    inColSize(0),
    size(0),
    groupCount(0),
    eps(1e-8),
    loading(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
GroupNorm<InputDataType, OutputDataType>::GroupNorm(
    const size_t inRowSize,
    const size_t inColSize,
    const size_t size,
    const size_t groupCount,
    const double eps):
    inRowSize(inRowSize),
    inColSize(inColSize),
    size(size),
    groupCount(groupCount),
    eps(eps),
    loading(false)
{
  if (inRowSize % groupCount != 0)
  {
    Log::Fatal << "Channel Size must be divisible by groupCount!" << std::endl;
  }

  weights.set_size(size * groupCount * 2 + 1, 1);
}

template<typename InputDataType, typename OutputDataType>
void GroupNorm<InputDataType, OutputDataType>::Reset()
{
  gamma = arma::mat(weights.memptr(), 1, groupCount * size, false, false);
  beta = arma::mat(weights.memptr() + gamma.n_elem, 1, groupCount * size, false, false);

  if (!loading)
  {
    gamma.fill(1.0);
    beta.fill(0.0);
  }

  loading = false;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  output = input;

  mean.set_size(1, groupCount * inColSize);
  variance.set_size(1, groupCount * inColSize);

  arma::cube outputAsCube((output).memptr(),
    inRowSize / groupCount, 1, groupCount*inColSize, false, false);

  for (int k = 0; k < groupCount * inColSize; ++k)
  {
    // Normalize the input.
    mean(k) = arma::mean(outputAsCube.slice(k));
    arma::mat var = arma::sqrt(arma::var(outputAsCube.slice(k), 0, 0) + eps);
    variance(k) = var(0);

    outputAsCube.slice(k) -= mean(k);
    outputAsCube.slice(k) /= variance(k);

    // Scale and shift the output.
    outputAsCube.slice(k) %= gamma(k);
    outputAsCube.slice(k) += beta(k);
  }

  // Reused in the backward and gradient step.
  normalized = output;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input, const arma::Mat<eT>& gradient, arma::Mat<eT>& output)
{
  if (output.is_empty())
    output.zeros(inRowSize * inColSize, size);
  else
  {
    assert(output.n_rows == inRowSize * inColSize);
    assert(output.n_cols == size);
  }

  arma::cube inputAsCube((input).memptr(),
    inRowSize / groupCount, 1, groupCount*inColSize, false, false);
  arma::cube outputAsCube((output).memptr(),
    inRowSize / groupCount, 1, groupCount*inColSize, false, false);
  arma::cube gradientAsCube((gradient).memptr(),
    inRowSize / groupCount, 1, groupCount*inColSize, false, false);

  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);
  for (int k = 0; k < groupCount * inColSize; ++k)
  {
    outputAsCube.slice(k) += gradientAsCube.slice(k) % gamma(k) % stdInv(k);
    outputAsCube.slice(k) += (gradientAsCube.slice(k) % (inputAsCube.slice(k) - mean(k))) % gamma(k) % 0.5 % std::pow(stdInv(k), 3.0) % (inputAsCube.slice(k) - mean(k)) % 2 / (inRowSize / groupCount);
    outputAsCube.slice(k) += gamma(k) % gradientAsCube.slice(k) / (inRowSize / groupCount);
  }

}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  gradient.set_size(size + size, 1);

  // Step 5: dl / dy * xhat.
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum(normalized % error, 1);

  // Step 6: dl / dy.
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(error, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void GroupNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(inRowSize));
  ar(CEREAL_NVP(inColSize));
  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(groupCount));

  if (cereal::is_loading<Archive>())
  {
    weights.set_size(size + size, 1);
    loading = true;
  }

  ar(CEREAL_NVP(eps));
  ar(CEREAL_NVP(gamma));
  ar(CEREAL_NVP(beta));
}

} // namespace ann
} // namespace mlpack

#endif
