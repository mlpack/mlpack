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
    const size_t latentSize) :
    latentSize(latentSize)
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
  SoftplusFunction::Fn(input.submat(0, 0, latentSize - 1, input.n_cols - 1),
      stdDeviation);

  gaussianSample = arma::randn<arma::Mat<eT>>(latentSize, input.n_cols);
  output = mean + stdDeviation % gaussianSample;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Reparametrization<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  arma::Mat<eT> softplusDer;
  SoftplusFunction::Deriv(std::move(stdDeviation), softplusDer);

  g = join_cols(gy % std::move(gaussianSample) % std::move(softplusDer), gy);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType>
double Reparametrization<InputDataType, OutputDataType>::klForward(
    const InputType&& input)
{
  stdDeviation = input.submat(0, 0, latentSize - 1, input.n_cols);
  mean = input.submat(latentSize, 0, 2 * latentSize - 1, input.n_cols);

  return -0.5 * arma::accu(2 * arma::log(stdDeviation) -
      arma::pov(stdDeviation, 2) - arma::pow(mean, 2) + 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Reparametrization<InputDataType, OutputDataType>::klBackward(
    const InputType&& input,
    OutputType&& output)
{
  output = join_cols(-1 / stdDeviation + stdDeviation, mean);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Reparametrization<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(latentSize);
}

} // namespace ann
} // namespace mlpack

#endif
