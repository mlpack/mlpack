/**
 * @file methods/ann/layer/threshold_fn_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of Threshold activation function
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_THRESHOLD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_THRESHOLD_IMPL_HPP

// In case it hasn't yet been included.
#include "threshold_fn.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Threshold<InputDataType, OutputDataType>::Threshold(
    const double threshold,
    const double value) : 
    threshold(threshold),
    value(value)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Threshold<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = arma::ones<OutputDataType>(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output(i) = (input(i) > threshold) ? input(i) : value;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void Threshold<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType derivative;
  derivative.set_size(arma::size(gy));
  derivative.fill(0.0);
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (input(i) > threshold)
      derivative(i) = 1.0;
  }

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Threshold<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(threshold));
  ar(CEREAL_NVP(value));
}

} // namespace ann
} // namespace mlpack

#endif
