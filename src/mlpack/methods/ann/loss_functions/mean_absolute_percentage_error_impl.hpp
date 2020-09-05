/**
 * @file methods/ann/loss_functions/mean_absolute_percentage_error_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of the Mean Absolute Percentage Error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_absolute_percentage_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MeanAbsolutePercentageError<InputDataType, OutputDataType>::MeanAbsolutePercentageError(
  const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MeanAbsolutePercentageError<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  InputType loss = arma::abs((input - target) / target);
  typename InputType::elem_type lossSum = arma::accu(loss);

  if (reduction)
    return lossSum;
    
  return lossSum / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MeanAbsolutePercentageError<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)

{ 
  if (input < target)
    output = - 1 / target;
  else 
    output = 1 / target;

  if (!reduction)
    output = output / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanAbsolutePercentageError<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(reduction);
}

} // namespace ann
} // namespace mlpack

#endif
