/**
 * @file methods/ann/loss_functions/l1_loss_impl.hpp
 * @author Himanshu Pathak
 *
 * Implementation of the L1 Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "l1_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
L1Loss<InputDataType, OutputDataType>::L1Loss(const bool mean):
  mean(mean)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
L1Loss<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  if (mean)
    return arma::accu(arma::mean(input - target));

  return arma::accu(input - target);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void L1Loss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output = arma::sign(input - target);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void L1Loss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(mean);
}

} // namespace ann
} // namespace mlpack

#endif
