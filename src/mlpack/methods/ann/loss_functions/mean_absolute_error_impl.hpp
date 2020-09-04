/**
 * @file methods/ann/loss_functions/mean_absolute_error_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of the Mean Absoulte Error function.
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
MeanAbsoluteError<InputDataType, OutputDataType>::MeanAbsoluteError(const bool mean):
  mean(mean)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MeanAbsoluteError<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  if (mean)
    return arma::accu(arma::mean(arma::abs((input - target) / input)));

  return arma::accu(arma::abs((input - target) / input));
}

template<typename InputDataType, typename OutputDataType>
template<InputType, typename TargetType, typename OutputType>
void MeanAbsoluteError<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const OutputType& output,
  OutputType& output)
{
  output = (target / arma::square(input)) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanAbsoluteError<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(mean);
}

} //namespace ann
} //namespace mlpack

#endif