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

template<typename MatType>
EarthMoverDistance<MatType>::EarthMoverDistance()
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type EarthMoverDistance<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  return -arma::accu(target % prediction);
}

template<typename MatType>
void EarthMoverDistance<MatType>::Backward(
    const MatType& /* prediction */,
    const MatType& target,
    MatType& loss)
{
  loss = -target;
}

template<typename MatType>
template<typename Archive>
void EarthMoverDistance<MatType>::serialize(
    Archive& /* ar */,
    const uint32_t /* version */)
{
  /* Nothing to do here */
}

} // namespace ann
} // namespace mlpack

#endif
