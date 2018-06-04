/**
 * @file sampling_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Sampling class which samples from parameters for a given
 * distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SAMPLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SAMPLING_IMPL_HPP

// In case it hasn't yet been included.
#include "sampling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Sampling<InputDataType, OutputDataType>::Sampling()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
Sampling<InputDataType, OutputDataType>::Sampling(
    const size_t inSize,
    const size_t outSize) :
    inSize(inSize),
    outSize(outSize)
{
  if (inSize != 2 * outSize)
  {
  	Log::Fatal << "The input size of Sampling layer should be 2 * output size!"
  	    << std::endl;
  }
}

template <typename InputDataType, typename OutputDataType>
Sampling<InputDataType, OutputDataType>::Sampling(
    const size_t sampleSize) :
    outSize(sampleSize)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sampling<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  if (input.n_rows != 2 * outSize)
  {
  	Log::Fatal << "The output size of layer before the Sampling layer should "
  	    << "be 2 * output size of the Sampling layer!" << std::endl;
  }

  arma::arma_rng::set_seed_random();

  mean = input.submat(outSize, 0, 2 * outSize - 1, input.n_cols - 1);
  SoftplusFunction::Fn(input.submat(0, 0, outSize - 1, input.n_cols - 1),
      stdDeviation);

  gaussianSample = arma::randn<arma::Mat<eT>>(outSize, input.n_cols);
  output = mean + stdDeviation % gaussianSample;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sampling<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{ 
  arma::Mat<eT> softplusDer;
  SoftplusFunction::Deriv((input - mean) / gaussianSample, 
      softplusDer);

  g = join_cols(gy % std::move(gaussianSample) % std::move(softplusDer), gy);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType>
double Sampling<InputDataType, OutputDataType>::klForward()
{
  return -0.5 * arma::accu(arma::log(stdDeviation) - stdDeviation -
      arma::pow(mean, 2) + 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Sampling<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  // ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);
}

} // namespace ann
} // namespace mlpack

#endif