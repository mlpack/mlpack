/**
 * @file methods/ann/layer/flatten_t_swish_impl.hpp
 * @author Fawwaz Mayda
 *
 * Definition of Flatten T Swish layer first introduced in the acoustic model,
 * Hock Hung Chieng, Noorhaniza Wahid, Pauline Ong, Sai Raj Kishore Perla,
 * "Flatten-T Swish: a thresholded ReLU-Swish-like activation function for deep learning", 2018
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_FLATTEN_T_SWISH_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_FLATTEN_T_SWISH_IMPL_HPP

// In case it hasn't yet been included.
#include "flatten_t_swish.hpp"
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

namespace mlpack {

template<typename InputDataType, typename OutputDataType>
FlattenTSwish<InputDataType, OutputDataType>::FlattenTSwish(
    const double T) : t(T)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void FlattenTSwish<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  // Placeholder for Relu values.
  OutputDataType relu;
  RectifierFunction::Fn(input, relu);
  LogisticFunction::Fn(input, output);
  // F(x) = relu * sigmoid + t.
  output = relu % output + t;
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void FlattenTSwish<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType derivate, sigmoid;
  LogisticFunction::Fn(input, sigmoid);
  derivate.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (input(i) >= 0)
    {
      // F(x) = x * sigmoid(x).
      // We don't put '+ t' here because this is a derivate.
      derivate(i) = input(i) * sigmoid(i);
      derivate(i) = sigmoid(i) * (1.0 - derivate(i)) + derivate(i);
    }
    else
    {
      derivate(i) = 0;
    }
  }
  g = gy % derivate;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void FlattenTSwish<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(t));
}

} // namespace mlpack

#endif
