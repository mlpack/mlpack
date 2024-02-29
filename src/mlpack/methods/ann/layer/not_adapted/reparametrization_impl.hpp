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

template <typename InputType, typename OutputType>
ReparametrizationType<InputType, OutputType>::ReparametrizationType(
    const bool stochastic,
    const bool includeKl,
    const double beta) :
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

template <typename InputType, typename OutputType>
ReparametrizationType<InputType, OutputType>::ReparametrizationType(
    const ReparametrizationType& layer) :
    Layer<InputType, OutputType>(layer),
    stochastic(layer.stochastic),
    includeKl(layer.includeKl),
    beta(layer.beta)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
ReparametrizationType<InputType, OutputType>::ReparametrizationType(
    ReparametrizationType&& layer) :
    Layer<InputType, OutputType>(std::move(layer)),
    stochastic(std::move(layer.stochastic)),
    includeKl(std::move(layer.includeKl)),
    beta(std::move(layer.beta))
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
ReparametrizationType<InputType, OutputType>&
ReparametrizationType<InputType, OutputType>::
operator=(const ReparametrizationType& layer)
{
  if (this != &layer)
  {
    Layer<InputType, OutputType>::operator=(layer);
    stochastic = layer.stochastic;
    includeKl = layer.includeKl;
    beta = layer.beta;
  }

  return *this;
}

template <typename InputType, typename OutputType>
ReparametrizationType<InputType, OutputType>&
ReparametrizationType<InputType, OutputType>::
operator=(ReparametrizationType&& layer)
{
  if (this != &layer)
  {
    Layer<InputType, OutputType>::operator=(std::move(layer));
    stochastic = std::move(layer.stochastic);
    includeKl = std::move(layer.includeKl);
    beta = std::move(layer.beta);
  }

  return *this;
}

template<typename InputType, typename OutputType>
void ReparametrizationType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  const size_t latentSize = this->outputDimensions[0];
  mean = input.submat(latentSize, 0, 2 * latentSize - 1, input.n_cols - 1);
  preStdDev = input.submat(0, 0, latentSize - 1, input.n_cols - 1);

  if (stochastic)
    gaussianSample.randn(latentSize, input.n_cols);
  else
    gaussianSample.ones(latentSize, input.n_cols) * 0.7;

  SoftplusFunction::Fn(preStdDev, stdDev);
  output = mean + stdDev % gaussianSample;
}

template<typename InputType, typename OutputType>
void ReparametrizationType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  OutputType tmp;
  SoftplusFunction::Deriv(preStdDev, stdDev, tmp);

  if (includeKl)
  {
    g = join_cols(gy % std::move(gaussianSample) % tmp + (-1 / stdDev + stdDev)
        % tmp * beta, gy + mean * beta / mean.n_cols);
  }
  else
  {
    g = join_cols(gy % std::move(gaussianSample) % tmp, gy);
  }
}

template<typename InputType, typename OutpuType>
double ReparametrizationType<InputType, OutpuType>::Loss()
{
  if (!includeKl)
    return 0;

  return -0.5 * beta * accu(2 * log(stdDev) - pow(stdDev, 2)
      - pow(mean, 2) + 1) / mean.n_cols;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void ReparametrizationType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(stochastic));
  ar(CEREAL_NVP(includeKl));
  ar(CEREAL_NVP(beta));
}

} // namespace mlpack

#endif
