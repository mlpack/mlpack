/**
 * @file core/tree/greedy_single_tree_traverser.hpp
 * @author Marcos Pividori
 *
 * A simple greedy traverser which always chooses the child with the best score
 * and doesn't do backtracking.  The RuleType class must implement the method
 * 'GetBestChild()'.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_GREEDY_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_GREEDY_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename TreeType, typename RuleType>
class GreedySingleTreeTraverser
{
 public:
  /**
   * Instantiate the greedy single tree traverser with the given rule set.
   */
  GreedySingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is being
   *     used as the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, TreeType& referenceNode);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }

 private:
  //! Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  //! The number of nodes which have been pruned during traversal.
  size_t numPrunes;
};

} // namespace mlpack

// Include implementation.
#include "greedy_single_tree_traverser_impl.hpp"

#endif
