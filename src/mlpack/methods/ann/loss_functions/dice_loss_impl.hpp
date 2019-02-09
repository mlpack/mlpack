/**
 * @file dice_loss_impl.hpp
 * @author N Rajiv Vaidyanathan
 *
 * Implementation of the dice loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "dice_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
DiceLoss<InputDataType, OutputDataType>::DiceLoss(
    const double smooth) : smooth(smooth)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double DiceLoss<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  return 1 - ((2 * arma::accu(target % input) + smooth) /
    (arma::accu(target % target) + arma::accu(
    input % input) + smooth));
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void DiceLoss<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  output = -2 * (target * (arma::accu(input % input) +
    arma::accu(target % target) + smooth) - input *
    (2 * arma::accu(target % input) + smooth)) / std::pow(
    arma::accu(target % target) + arma::accu(input % input)
    + smooth, 2.0);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void DiceLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(smooth);
}

} // namespace ann
} // namespace mlpack

#endif
