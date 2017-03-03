/**
 * @file traversal_info.hpp
 * @author Ryan Curtin
 *
 * This class will hold the traversal information for dual-tree traversals.  A
 * dual-tree traversal should be updating the members of this class before
 * Score() is called.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_TRAVERSAL_INFO_HPP
#define MLPACK_CORE_TREE_TRAVERSAL_INFO_HPP

namespace mlpack {
namespace tree {

/**
 * The TraversalInfo class holds traversal information which is used in
 * dual-tree (and single-tree) traversals.  A traversal should be updating the
 * members of this class before Score() is called.  This class should be held as
 * a member of the RuleType class and the interface to it should be through a
 * TraversalInfo() method.
 *
 * The information held by this class is the last node combination visited
 * before the current node combination was recursed into, and the score
 * resulting from when Score() was called on that combination.  However, this
 * information is identical for a query node and a reference node in a
 * particular node combination, so traversals only need to update the
 * TraversalInfo object in a query node (and the algorithms should only use the
 * TraversalInfo object from a query node).
 *
 * In general, this auxiliary traversal information is used to try and make a
 * prune without needing to call BaseCase() or calculate the distance between
 * nodes.  Using this information you can place bounds on the distance between
 * the two nodes quickly.
 *
 * If the traversal is not updating the members of this class correctly, a
 * likely result is a null pointer dereference.  Dual-tree algorithms should
 * assume that the members are set properly and should not need to check for
 * null pointers.
 *
 * There is one exception, which is the root node combination; the score can be
 * set to 0 and the query and reference nodes can just be set to the root nodes;
 * no algorithm should be able to prune the root combination anyway.
 */
template<typename TreeType>
class TraversalInfo
{
 public:
  /**
   * Create the TraversalInfo object and initialize the pointers to NULL.
   */
  TraversalInfo() :
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
  //! The last score.
  double lastScore;
  //! The last base case.
  double lastBaseCase;
};

} // namespace tree
} // namespace mlpack

#endif
