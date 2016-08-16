/**
 * @file midpoint_space_split_impl.hpp
 * @author Marcos Pividori
 *
 * Implementation of MidpointSpaceSplit, to create a splitting hyperplane
 * considering the midpoint of the values in a certain projection.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_MIDPOINT_SPACE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_MIDPOINT_SPACE_SPLIT_IMPL_HPP

#include "midpoint_space_split.hpp"
#include "space_split.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType, typename MatType>
template<typename HyperplaneType>
bool MidpointSpaceSplit<MetricType, MatType>::SplitSpace(
    const typename HyperplaneType::BoundType& bound,
    const MatType& data,
    const arma::Col<size_t>& points,
    HyperplaneType& hyp)
{
  typename HyperplaneType::ProjVectorType projVector;
  double midValue;

  if (!SpaceSplit<MetricType, MatType>::GetProjVector(bound, data, points,
      projVector, midValue))
    return false;

  hyp = HyperplaneType(projVector, midValue);

  return true;
}

} // namespace tree
} // namespace mlpack

#endif
