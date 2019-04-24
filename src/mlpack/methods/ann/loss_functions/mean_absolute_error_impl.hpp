/**
 * @file mean_absolute_error_impl.hpp
 * @author Abhinav Sagar
 *
 * Implementation of the mean absolute error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_absolute_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MeanAbsoluteError<InputDataType, OutputDataType>::MeanAbsoluteError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double MeanAbsoluteError<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  return arma::accu(arma::abs(input - target)) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MeanAbsoluteError<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  output = arma::abs(input - target) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanAbsoluteError<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
