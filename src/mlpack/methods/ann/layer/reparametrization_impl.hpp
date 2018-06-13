/**
 * @file reparametrization_impl.hpp
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
Reparametrization<InputDataType, OutputDataType>::Reparametrization()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
Reparametrization<InputDataType, OutputDataType>::Reparametrization(
    const size_t latentSize,
    const bool stochastic,
    const bool includeKl) :
    latentSize(latentSize),
    stochastic(stochastic),
    includeKl(includeKl)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Reparametrization<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
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
    gaussianSample = arma::randn<arma::Mat<eT>>(latentSize, input.n_cols);
  else
    gaussianSample = arma::ones<arma::Mat<eT>>(latentSize, input.n_cols) * 0.7;

  SoftplusFunction::Fn(preStdDev, stdDev);
  output = mean + stdDev % gaussianSample;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Reparametrization<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  SoftplusFunction::Deriv(preStdDev, g);

  if (includeKl)
  {
    arma::Mat<eT> klBack;
    klBackward(std::move(klBack));
    g = join_cols(gy % std::move(gaussianSample) % g, gy) + std::move(klBack);
  }
  else
    g = join_cols(gy % std::move(gaussianSample) % g, gy);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType>
double Reparametrization<InputDataType, OutputDataType>::klForward(
    const InputType&& input)
{
  stdDev = input.submat(0, 0, latentSize - 1, input.n_cols - 1);
  mean = input.submat(latentSize, 0, 2 * latentSize - 1, input.n_cols - 1);

  return -0.5 * arma::accu(2 * arma::log(stdDev) -
      arma::pow(stdDev, 2) - arma::pow(mean, 2) + 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename OutputType>
void Reparametrization<InputDataType, OutputDataType>::klBackward(
    OutputType&& output)
{
  SoftplusFunction::Deriv(preStdDev, output);
  output = join_cols((-1 / stdDev + stdDev) % output, mean);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Reparametrization<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(latentSize);
  ar & BOOST_SERIALIZATION_NVP(stochastic);
}

} // namespace ann
} // namespace mlpack

#endif
