/**
 * @file neighbor_search.hpp
 * @author Ryan Curtin
 *
 * Defines the NeighborSearch class, which performs an abstract
 * nearest-neighbor-like query on two datasets.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_STAT_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_STAT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

/**
 * Extra data for each node in the tree.  For neighbor searches, each node only
 * needs to store a bound on neighbor distances.
 */
template<typename SortPolicy>
class NeighborSearchStat
{
 private:
  //! The first bound on the node's neighbor distances (B_1).  This represents
  //! the worst candidate distance of any descendants of this node.
  double firstBound;
  //! The second bound on the node's neighbor distances (B_2).  This represents
  //! a bound on the worst distance of any descendants of this node assembled
  //! using the best descendant candidate distance modified by the furthest
  //! descendant distance.
  double secondBound;
  //! The better of the two bounds.
  double bound;

  //! The last distance evaluation node.
  void* lastDistanceNode;
  //! The last distance evaluation.
  double lastDistance;

 public:
  /**
   * Initialize the statistic with the worst possible distance according to
   * our sorting policy.
   */
  NeighborSearchStat() :
      firstBound(SortPolicy::WorstDistance()),
      secondBound(SortPolicy::WorstDistance()),
      bound(SortPolicy::WorstDistance()),
      lastDistanceNode(NULL),
      lastDistance(0.0) { }

  /**
   * Initialization for a fully initialized node.  In this case, we don't need
   * to worry about the node.
   */
  template<typename TreeType>
  NeighborSearchStat(TreeType& /* node */) :
      firstBound(SortPolicy::WorstDistance()),
      secondBound(SortPolicy::WorstDistance()),
      bound(SortPolicy::WorstDistance()),
      lastDistanceNode(NULL),
      lastDistance(0.0) { }

  //! Get the first bound.
  double FirstBound() const { return firstBound; }
  //! Modify the first bound.
  double& FirstBound() { return firstBound; }
  //! Get the second bound.
  double SecondBound() const { return secondBound; }
  //! Modify the second bound.
  double& SecondBound() { return secondBound; }
  //! Get the overall bound (the better of the two bounds).
  double Bound() const { return bound; }
  //! Modify the overall bound (it should be the better of the two bounds).
  double& Bound() { return bound; }
  //! Get the last distance evaluation node.
  void* LastDistanceNode() const { return lastDistanceNode; }
  //! Modify the last distance evaluation node.
  void*& LastDistanceNode() { return lastDistanceNode; }
  //! Get the last distance calculation.
  double LastDistance() const { return lastDistance; }
  //! Modify the last distance calculation.
  double& LastDistance() { return lastDistance; }
};

}; // namespace neighbor
}; // namespace mlpack

#endif
