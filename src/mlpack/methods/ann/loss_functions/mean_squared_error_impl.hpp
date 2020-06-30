/**
 * @file methods/ann/loss_functions/mean_squared_error_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the mean squared error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_squared_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MeanSquaredError<InputDataType, OutputDataType>::MeanSquaredError(
    const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MeanSquaredError<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  InputType loss = arma::square(input - target);
  typename InputType::elem_type lossSum = arma::accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / input.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MeanSquaredError<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output = 2 * (input - target);

  if (!reduction)
    output = output / input.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanSquaredError<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(reduction);
}

} // namespace ann
} // namespace mlpack

#endif
