/**
 * @file mean_bias_error_impl.hpp
 * @author Saksham Rastogi
 *
 * Implementation of the mean bias error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_BIAS_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_BIAS_ERROR_IMPL_HPP


// In case it hasn't yet been included.
#include "mean_bias_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MeanBiasError<InputDataType, OutputDataType>::MeanBiasError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double MeanBiasError<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  return arma::accu(target - input) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MeanBiasError<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  output.set_size(arma::size(input));
  output.fill(-1.0);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanBiasError<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
