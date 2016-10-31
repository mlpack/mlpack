/***
 * @file furthest_neighbor_sort_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated methods for the FurthestNeighborSort SortPolicy
 * class for the NeighborSearch class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP

namespace mlpack {
namespace neighbor {

template<typename TreeType>
inline double FurthestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode)
{
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return queryNode->MaxDistance(*referenceNode);
}

template<typename TreeType>
inline double FurthestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode,
    const double centerToCenterDistance)
{
  return queryNode->MaxDistance(*referenceNode, centerToCenterDistance);
}

template<typename TreeType>
inline double FurthestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode,
    const TreeType* referenceChildNode,
    const double centerToCenterDistance)
{
  return queryNode->MaxDistance(*referenceNode, centerToCenterDistance) +
      referenceChildNode->ParentDistance();
}

template<typename VecType, typename TreeType>
inline double FurthestNeighborSort::BestPointToNodeDistance(
    const VecType& point,
    const TreeType* referenceNode)
{
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return referenceNode->MaxDistance(point);
}

template<typename VecType, typename TreeType>
inline double FurthestNeighborSort::BestPointToNodeDistance(
    const VecType& point,
    const TreeType* referenceNode,
    const double pointToCenterDistance)
{
  return referenceNode->MaxDistance(point, pointToCenterDistance);
}

} // namespace neighbor
} // namespace mlpack

#endif
