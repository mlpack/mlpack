/**
 * @file nearest_neighbor_sort_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated methods for the NearestNeighborSort SortPolicy
 * class for the NeighborSearch class.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_IMPL_HPP
#define __MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_IMPL_HPP

namespace mlpack {
namespace neighbor {

template<typename TreeType>
inline double NearestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode)
{
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return queryNode->MinDistance(referenceNode);
}

template<typename TreeType>
inline double NearestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode,
    const double centerToCenterDistance)
{
  return queryNode->MinDistance(referenceNode, centerToCenterDistance);
}

template<typename TreeType>
inline double NearestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* /* referenceNode */,
    const TreeType* referenceChildNode,
    const double centerToCenterDistance)
{
  return queryNode->MinDistance(referenceChildNode, centerToCenterDistance) -
      referenceChildNode->ParentDistance();
}

template<typename VecType, typename TreeType>
inline double NearestNeighborSort::BestPointToNodeDistance(
    const VecType& point,
    const TreeType* referenceNode)
{
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return referenceNode->MinDistance(point);
}

template<typename VecType, typename TreeType>
inline double NearestNeighborSort::BestPointToNodeDistance(
    const VecType& point,
    const TreeType* referenceNode,
    const double pointToCenterDistance)
{
  return referenceNode->MinDistance(point, pointToCenterDistance);
}

}; // namespace neighbor
}; // namespace mlpack

#endif
