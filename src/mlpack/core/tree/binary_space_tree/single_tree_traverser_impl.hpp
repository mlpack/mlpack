/**
 * @file single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * A nested class of BinarySpaceTree which traverses the entire tree with a
 * given set of rules which indicate the branches which can be pruned and the
 * order in which to recurse.  This traverser is a depth-first traverser.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

#include <stack>

namespace mlpack {
namespace tree {

template<typename BoundType, typename StatisticType, typename MatType>
template<typename RuleType>
BinarySpaceTree<BoundType, StatisticType, MatType>::
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename BoundType, typename StatisticType, typename MatType>
template<typename RuleType>
void BinarySpaceTree<BoundType, StatisticType, MatType>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    BinarySpaceTree<BoundType, StatisticType, MatType>& referenceNode)
{
  // This is a non-recursive implementation (which should be faster).

  // The stack of points to look at.
  std::stack<BinarySpaceTree<BoundType, StatisticType, MatType>*> pointStack;
  pointStack.push(&referenceNode);

  while (!pointStack.empty())
  {
    BinarySpaceTree<BoundType, StatisticType, MatType>* node = pointStack.top();
    pointStack.pop();

    // Check if we can prune this node.
    if (rule.CanPrune(queryIndex, *node))
    {
      ++numPrunes;
      continue;
    }

    // If we are a leaf, run the base case as necessary.
    if (node->IsLeaf())
    {
      for (size_t i = node->Begin(); i < node->End(); ++i)
        rule.BaseCase(queryIndex, i);
    }
    else
    {
      // Otherwise recurse by distance.
      if (rule.LeftFirst(queryIndex, *node))
      {
        pointStack.push(node->Right());
        pointStack.push(node->Left());
      }
      else
      {
        pointStack.push(node->Left());
        pointStack.push(node->Right());
      }
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
