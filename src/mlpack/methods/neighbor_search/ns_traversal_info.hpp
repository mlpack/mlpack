/**
 * @file ns_traversal_info.hpp
 * @author Ryan Curtin
 *
 * This class holds traversal information for dual-tree traversals that are
 * using the NeighborSearchRules RuleType.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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

} // namespace neighbor
} // namespace mlpack

#endif
