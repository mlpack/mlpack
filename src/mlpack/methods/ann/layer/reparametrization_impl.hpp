/**
 * @file methods/ann/layer/reparametrization_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Reparametrization layer class which samples from a
 * gaussian distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPARAMETRIZATION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_REPARAMETRIZATION_IMPL_HPP

// In case it hasn't yet been included.
#include "reparametrization.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
ReparametrizationType<InputType, OutputType>::ReparametrizationType() :
    latentSize(0),
    stochastic(true),
    includeKl(true),
    beta(1)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
ReparametrizationType<InputType, OutputType>::ReparametrizationType(
    const size_t latentSize,
    const bool stochastic,
    const bool includeKl,
    const double beta) :
    latentSize(latentSize),
    stochastic(stochastic),
    includeKl(includeKl),
    beta(beta)
{
  if (includeKl == false && beta != 1)
  {
    Log::Info << "The beta parameter will be ignored as KL divergence is not "
        << "included." << std::endl;
  }
}

template<typename InputType, typename OutputType>
void ReparametrizationType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (input.n_rows != 2 * latentSize)
  {
    Log::Fatal << "The output size of layer before the Reparametrization "
        << "layer should be 2 * latent size of the Reparametrization layer!"
        << std::endl;
  }

  mean = input.submat(latentSize, 0, 2 * latentSize - 1, input.n_cols - 1);
  preStdDev = input.submat(0, 0, latentSize - 1, input.n_cols - 1);

  if (stochastic)
    gaussianSample = arma::randn<OutputType>(latentSize, input.n_cols);
  else
    gaussianSample = arma::ones<OutputType>(latentSize, input.n_cols) * 0.7;

  SoftplusFunction::Fn(preStdDev, stdDev);
  output = mean + stdDev % gaussianSample;
}

template<typename InputType, typename OutputType>
void ReparametrizationType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  SoftplusFunction::Deriv(preStdDev, g);

  if (includeKl)
  {
    g = join_cols(gy % std::move(gaussianSample) % g + (-1 / stdDev + stdDev)
        % g * beta, gy + mean * beta / mean.n_cols);
  }
  else
    g = join_cols(gy % std::move(gaussianSample) % g, gy);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void ReparametrizationType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(latentSize));
  ar(CEREAL_NVP(stochastic));
  ar(CEREAL_NVP(includeKl));
}

} // namespace ann
} // namespace mlpack

#endif
