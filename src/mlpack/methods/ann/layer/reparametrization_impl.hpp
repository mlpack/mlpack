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

template<typename InputDataType, typename OutputDataType>
Reparametrization<InputDataType, OutputDataType>::Reparametrization() :
    latentSize(0),
    stochastic(true),
    includeKl(true),
    beta(1)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
Reparametrization<InputDataType, OutputDataType>::Reparametrization(
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Reparametrization<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
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
    gaussianSample = arma::randn<arma::Mat<eT> >(latentSize, input.n_cols);
  else
    gaussianSample = arma::ones<arma::Mat<eT> >(latentSize, input.n_cols) * 0.7;

  SoftplusFunction::Fn(preStdDev, stdDev);
  output = mean + stdDev % gaussianSample;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Reparametrization<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
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

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Reparametrization<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(latentSize);
  ar & CEREAL_NVP(stochastic);
  ar & CEREAL_NVP(includeKl);
}

} // namespace ann
} // namespace mlpack

#endif
