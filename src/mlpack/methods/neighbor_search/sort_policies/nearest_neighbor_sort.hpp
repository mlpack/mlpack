/**
 * @file nearest_neighbor_sort.hpp
 * @author Ryan Curtin
 *
 * Implementation of the SortPolicy class for NeighborSearch; in this case, the
 * nearest neighbors are those that are most important.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_SORT_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_SORT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

/**
 * This class implements the necessary methods for the SortPolicy template
 * parameter of the NeighborSearch class.  The sorting policy here is that the
 * minimum distance is the best (so, when used with NeighborSearch, the output
 * is nearest neighbors).
 *
 * This class is also meant to serve as a guide to implement a custom
 * SortPolicy.  All of the methods implemented here must be implemented by any
 * other SortPolicy classes.
 */
class NearestNeighborSort
{
 public:
  /**
   * Return the index in the vector where the new distance should be inserted,
   * or (size_t() - 1) if it should not be inserted (i.e. if it is not any
   * better than any of the existing points in the list).  The list should be
   * sorted such that the best point is the first in the list.  The actual
   * insertion is not performed.
   *
   * @param list Vector of existing distance points, sorted such that the best
   *     point is first in the list.
   * @param new_distance Distance to try to insert
   *
   * @return size_t containing the position to insert into, or (size_t() - 1)
   *     if the new distance should not be inserted.
   */
  static size_t SortDistance(const arma::vec& list,
                             const arma::Col<size_t>& indices,
                             double newDistance);

  /**
   * Return whether or not value is "better" than ref.  In this case, that means
   * that the value is less than the reference.
   *
   * @param value Value to compare
   * @param ref Value to compare with
   *
   * @return bool indicating whether or not (value < ref).
   */
  static inline bool IsBetter(const double value, const double ref)
  {
    return (value < ref);
  }

  /**
   * Return the best possible distance between two nodes.  In our case, this is
   * the minimum distance between the two tree nodes using the given distance
   * function.
   */
  template<typename TreeType>
  static double BestNodeToNodeDistance(const TreeType* queryNode,
                                       const TreeType* referenceNode);

  /**
   * Return the best possible distance between two nodes, given that the
   * distance between the centers of the two nodes has already been calculated.
   * This is used in conjunction with trees that have self-children (like cover
   * trees).
   */
  template<typename TreeType>
  static double BestNodeToNodeDistance(const TreeType* queryNode,
                                       const TreeType* referenceNode,
                                       const double centerToCenterDistance);

  /**
   * Return the best possible distance between the query node and the reference
   * child node given the base case distance between the query node and the
   * reference node.  TreeType::ParentDistance() must be implemented to use
   * this.
   *
   * @param queryNode Query node.
   * @param referenceNode Reference node.
   * @param referenceChildNode Child of reference node which is being inspected.
   * @param centerToCenterDistance Distance between centers of query node and
   *     reference node.
   */
  template<typename TreeType>
  static double BestNodeToNodeDistance(const TreeType* queryNode,
                                       const TreeType* referenceNode,
                                       const TreeType* referenceChildNode,
                                       const double centerToCenterDistance);
  /**
   * Return the best possible distance between a node and a point.  In our case,
   * this is the minimum distance between the tree node and the point using the
   * given distance function.
   */
  template<typename VecType, typename TreeType>
  static double BestPointToNodeDistance(const VecType& queryPoint,
                                        const TreeType* referenceNode);

  /**
   * Return the best possible distance between a point and a node, given that
   * the distance between the point and the center of the node has already been
   * calculated.  This is used in conjunction with trees that have
   * self-children (like cover trees).
   */
  template<typename VecType, typename TreeType>
  static double BestPointToNodeDistance(const VecType& queryPoint,
                                        const TreeType* referenceNode,
                                        const double pointToCenterDistance);

  /**
   * Return what should represent the worst possible distance with this
   * particular sort policy.  In our case, this should be the maximum possible
   * distance, DBL_MAX.
   *
   * @return DBL_MAX
   */
  static inline double WorstDistance() { return DBL_MAX; }

  /**
   * Return what should represent the best possible distance with this
   * particular sort policy.  In our case, this should be the minimum possible
   * distance, 0.0.
   *
   * @return 0.0
   */
  static inline double BestDistance() { return 0.0; }

  /**
   * Return the best combination of the two distances.
   */
  static inline double CombineBest(const double a, const double b)
  {
    return std::max(a - b, 0.0);
  }

  /**
   * Return the worst combination of the two distances.
   */
  static inline double CombineWorst(const double a, const double b)
  {
    if (a == DBL_MAX || b == DBL_MAX)
      return DBL_MAX;
    return a + b;
  }
};

}; // namespace neighbor
}; // namespace mlpack

// Include implementation of templated functions.
#include "nearest_neighbor_sort_impl.hpp"

#endif
