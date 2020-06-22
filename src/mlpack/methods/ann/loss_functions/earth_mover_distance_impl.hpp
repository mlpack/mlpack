/**
 * @file methods/ann/loss_functions/earth_mover_distance_impl.hpp
 * @author Shikhar Jaiswal
 *
 * Implementation of the earth mover distance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_IMPL_HPP

// In case it hasn't yet been included.
#include "earth_mover_distance.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
EarthMoverDistance<InputDataType, OutputDataType>::EarthMoverDistance()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
EarthMoverDistance<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  return -arma::accu(target % input);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void EarthMoverDistance<InputDataType, OutputDataType>::Backward(
    const InputType& /* input */,
    const TargetType& target,
    OutputType& output)
{
  output = -target;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void EarthMoverDistance<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  /* Nothing to do here */
}

} // namespace ann
} // namespace mlpack

#endif
