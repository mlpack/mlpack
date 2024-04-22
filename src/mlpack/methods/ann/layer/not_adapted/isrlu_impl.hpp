/**
 * @file methods/ann/layer/isrlu_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the ISRLU activation function as described by Jonathan T. Barron.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ISRLU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ISRLU_IMPL_HPP

// In case it hasn't yet been included.
#include "isrlu.hpp"

namespace mlpack {

template<typename InputDataType, typename OutputDataType>
ISRLU<InputDataType, OutputDataType>::ISRLU(const double alpha) :
    alpha(alpha)
{}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void ISRLU<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = ones<OutputDataType>(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output(i) = (input(i) >= 0) ? input(i) : input(i) *
        (1 / std::sqrt(1 + alpha * (input(i) * input(i))));
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void ISRLU<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  derivative.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    derivative(i) = (input(i) >= 0) ? 1 :
        std::pow(1 / std::sqrt(1 + alpha * input(i) * input(i)), 3);
  }
  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ISRLU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif
