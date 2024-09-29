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

template<typename MatType>
EarthMoverDistanceType<MatType>::EarthMoverDistanceType(const bool reduction) :
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type EarthMoverDistanceType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  typename MatType::elem_type lossSum = -accu(target % prediction);

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void EarthMoverDistanceType<MatType>::Backward(
    const MatType& /* prediction */,
    const MatType& target,
    MatType& loss)
{
  loss = -target;

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename MatType>
template<typename Archive>
void EarthMoverDistanceType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
