/**
 * @file methods/ann/loss_functions/mean_squared_logarithmic_error_impl.hpp
 * @author Saksham Rastogi
 *
 * Implementation of the mean squared logarithmic error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_LOGARITHMIC_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_LOGARITHMIC_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_squared_logarithmic_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MeanSquaredLogarithmicError<InputDataType, OutputDataType>
::MeanSquaredLogarithmicError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MeanSquaredLogarithmicError<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  return arma::accu(arma::square(arma::log(1. + target) -
      arma::log(1. + input))) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MeanSquaredLogarithmicError<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output = 2 * (arma::log(1. + input) - arma::log(1. + target)) /
      ((1. + input) * target.n_cols);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanSquaredLogarithmicError<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
