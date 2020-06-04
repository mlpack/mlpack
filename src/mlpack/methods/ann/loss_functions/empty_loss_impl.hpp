/**
 * @file methods/ann/loss_functions/empty_loss_impl.hpp
 * @author Xiaohong Ji
 *
 * Implementation of empty loss function. Sometimes, the user may want to
 * calculate the loss outside of the model, so we have created an empty loss
 * that does nothing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EMPTY_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EMPTY_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "empty_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
EmptyLoss<InputDataType, OutputDataType>::EmptyLoss()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double EmptyLoss<InputDataType, OutputDataType>::Forward(
    const InputType& /* input */, const TargetType& /* target */)
{
  return 0;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void EmptyLoss<InputDataType, OutputDataType>::Backward(
    const InputType& /* input */,
    const TargetType& target,
    OutputType& output)
{
  output = target;
}

} // namespace ann
} // namespace mlpack

#endif
