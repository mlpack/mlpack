/**
 * @file greedy_single_tree_traverser_impl.hpp
 * @author Marcos Pividori
 *
 * A simple greedy traverser which always chooses the child with the best score
 * and doesn't do backtracking.  The RuleType class must implement the method
 * 'GetBestChild()'.
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
  // If we have reached a leaf node, run the base case as necessary.
  if (referenceNode.IsLeaf())
  {
    for (size_t i = 0; i < referenceNode.NumPoints(); ++i)
      rule.BaseCase(queryIndex, referenceNode.Point(i));
  }
  else
  {
    // We are prunning all but one child.
    numPrunes += referenceNode.NumChildren() - 1;
    // Recurse the best child.
    TreeType& bestChild = rule.GetBestChild(queryIndex, referenceNode);
    Traverse(queryIndex, bestChild);
  }
}

} // namespace tree
} // namespace mlpack

#endif
