/***
 * @file furthest_neighbor_sort_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated methods for the FurthestNeighborSort SortPolicy
 * class for the NeighborSearch class.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP

namespace mlpack {
namespace neighbor {

template<typename TreeType>
inline double FurthestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode)
{
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return queryNode->MaxDistance(referenceNode);
}

template<typename TreeType>
inline double FurthestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode,
    const double centerToCenterDistance)
{
  return queryNode->MaxDistance(referenceNode, centerToCenterDistance);
}

template<typename TreeType>
inline double FurthestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode,
    const TreeType* referenceChildNode,
    const double centerToCenterDistance)
{
  return queryNode->MaxDistance(referenceNode, centerToCenterDistance) +
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

}; // namespace neighbor
}; // namespace mlpack

#endif
