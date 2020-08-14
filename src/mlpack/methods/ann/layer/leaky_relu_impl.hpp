/**
 * @file methods/ann/layer/leaky_relu_impl.hpp
 * @author Dhawal Arora
 *
 * Implementation of LeakyReLU layer first introduced in the acoustic model,
 * Andrew L. Maas, Awni Y. Hannun, Andrew Y. Ng,
 * "Rectifier Nonlinearities Improve Neural Network Acoustic Models", 2014
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LEAKYRELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LEAKYRELU_IMPL_HPP

// In case it hasn't yet been included.
#include "leaky_relu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
LeakyReLU<InputDataType, OutputDataType>::LeakyReLU(
    const double alpha) : alpha(alpha)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void LeakyReLU<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = arma::max(input, alpha * input);
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void LeakyReLU<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType derivative;
  derivative.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
    derivative(i) = (input(i) >= 0) ? 1 : alpha;

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void LeakyReLU<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(alpha);
}

} // namespace ann
} // namespace mlpack

#endif
