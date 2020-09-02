/**
 * @file methods/ann/loss_functions/sigmoid_cross_entropy_error_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * Implementation of the sigmoid cross entropy error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "sigmoid_cross_entropy_error.hpp"
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SigmoidCrossEntropyError<InputDataType, OutputDataType>
::SigmoidCrossEntropyError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
inline typename InputType::elem_type
SigmoidCrossEntropyError<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  typedef typename InputType::elem_type ElemType;
  ElemType maximum = 0;
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    maximum += std::max(input[i], 0.0) +
        std::log(1 + std::exp(-std::abs(input[i])));
  }

  return maximum - arma::accu(input % target);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
inline void SigmoidCrossEntropyError<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output = 1.0 / (1.0 + arma::exp(-input)) - target;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SigmoidCrossEntropyError<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here
}

} // namespace ann
} // namespace mlpack

#endif
