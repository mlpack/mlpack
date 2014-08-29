/**
 * @file ns_traversal_info.hpp
 * @author Ryan Curtin
 *
 * This class holds traversal information for dual-tree traversals that are
 * using the NeighborSearchRules RuleType.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_TRAVERSAL_INFO_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_TRAVERSAL_INFO_HPP

namespace mlpack {
namespace neighbor {

/**
 * Traversal information for NeighborSearch.  This information is used to make
 * parent-child prunes or parent-parent prunes in Score() without needing to
 * evaluate the distance between two nodes.
 *
 * The information held by this class is the last node combination visited
 * before the current node combination was recursed into and the distance
 * between the node centroids.
 */
template<typename TreeType>
class NeighborSearchTraversalInfo
{
 public:
  /**
   * Create the TraversalInfo object and initialize the pointers to NULL.
   */
  NeighborSearchTraversalInfo() :
      lastQueryNode(NULL),
      lastReferenceNode(NULL),
      lastScore(0.0),
      lastBaseCase(0.0) { /* Nothing to do. */ }

   //! Get the last query node.
  TreeType* LastQueryNode() const { return lastQueryNode; }
  //! Modify the last query node.
  TreeType*& LastQueryNode() { return lastQueryNode; }

  //! Get the last reference node.
  TreeType* LastReferenceNode() const { return lastReferenceNode; }
  //! Modify the last reference node.
  TreeType*& LastReferenceNode() { return lastReferenceNode; }

  //! Get the score associated with the last query and reference nodes.
  double LastScore() const { return lastScore; }
  //! Modify the score associated with the last query and reference nodes.
  double& LastScore() { return lastScore; }

  //! Get the base case associated with the last node combination.
  double LastBaseCase() const { return lastBaseCase; }
  //! Modify the base case associated with the last node combination.
  double& LastBaseCase() { return lastBaseCase; }

 private:
  //! The last query node.
  TreeType* lastQueryNode;
  //! The last reference node.
  TreeType* lastReferenceNode;
  //! The last distance.
  double lastScore;
  //! The last base case.
  double lastBaseCase;
};

}; // namespace neighbor
}; // namespace mlpack

#endif
