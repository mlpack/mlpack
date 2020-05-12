/**
 * @file Empty_loss_impl.hpp
 * @author Xiaohong Ji
 *
 * Definition of empty loss function. Sometime, user may want to calculate the
 * loss outside of the model, so we create an empty loss that do nothing.
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
  const InputType& input, const TargetType& target)
{
  return arma::accu(target) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void EmptyLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  // todo: check correctness
  output = target / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void EmptyLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
