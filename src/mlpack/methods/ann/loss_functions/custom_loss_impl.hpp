/**
 * @file custom_loss_impl.hpp
 * @author Prince Gupta
 *
 * Implementation of the mean squared error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_CUSTOM_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_CUSTOM_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "custom_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
CustomLoss<InputDataType, OutputDataType>::CustomLoss(
    std::function<double(const InputDataType&&, const InputDataType&&)> forward,
    std::function<void(const InputDataType&&,
                       const InputDataType&&,
                       OutputDataType&&)> backward) :
      forward(forward), backward(backward)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double CustomLoss<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  return forward(std::move(input), std::move(target));
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void CustomLoss<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  backward(std::move(input), std::move(target), std::move(output));
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void CustomLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
