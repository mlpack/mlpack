/**
 * @file greedy_single_tree_traverser_impl.hpp
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
#ifndef MLPACK_CORE_TREE_GREEDY_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_GREEDY_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "greedy_single_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType, typename RuleType>
GreedySingleTreeTraverser<TreeType, RuleType>::GreedySingleTreeTraverser(
    RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename TreeType, typename RuleType>
void GreedySingleTreeTraverser<TreeType, RuleType>::Traverse(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  // Run the base case as necessary for all the points in the reference node.
  for (size_t i = 0; i < referenceNode.NumPoints(); ++i)
    rule.BaseCase(queryIndex, referenceNode.Point(i));

  if (!referenceNode.IsLeaf())
  {
    // We are prunning all but one child.
    numPrunes += referenceNode.NumChildren() - 1;
    // Recurse the best child.
    size_t bestChild = rule.GetBestChild(queryIndex, referenceNode);
    Traverse(queryIndex, referenceNode.Child(bestChild));
  }
}

} // namespace tree
} // namespace mlpack

#endif
