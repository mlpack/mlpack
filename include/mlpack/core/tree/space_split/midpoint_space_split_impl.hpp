/**
 * @file core/tree/space_split/midpoint_space_split_impl.hpp
 * @author Marcos Pividori
 *
 * Implementation of MidpointSpaceSplit, to create a splitting hyperplane
 * considering the midpoint of the values in a certain projection.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_MIDPOINT_SPACE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_MIDPOINT_SPACE_SPLIT_IMPL_HPP

#include "midpoint_space_split.hpp"
#include "space_split.hpp"

namespace mlpack {

template<typename DistanceType, typename MatType>
template<typename HyperplaneType>
bool MidpointSpaceSplit<DistanceType, MatType>::SplitSpace(
    const typename HyperplaneType::BoundType& bound,
    const MatType& data,
    const arma::Col<size_t>& points,
    HyperplaneType& hyp)
{
  typename HyperplaneType::ProjVectorType projVector;
  double midValue;

  if (!SpaceSplit<DistanceType, MatType>::GetProjVector(bound, data, points,
      projVector, midValue))
    return false;

  hyp = HyperplaneType(projVector, midValue);

  return true;
}

} // namespace mlpack

#endif
