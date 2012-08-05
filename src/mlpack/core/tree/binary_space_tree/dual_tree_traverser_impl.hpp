/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the DualTreeTraverser
 * way to perform a dual-tree traversal of two trees.  The trees must be the
 * same type.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename BoundType, typename StatisticType, typename MatType>
template<typename RuleType>
BinarySpaceTree<BoundType, StatisticType, MatType>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename BoundType, typename StatisticType, typename MatType>
template<typename RuleType>
void BinarySpaceTree<BoundType, StatisticType, MatType>::
DualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<BoundType, StatisticType, MatType>& queryNode,
    BinarySpaceTree<BoundType, StatisticType, MatType>& referenceNode)
{
  // Check if pruning can occur.
  if (rule.CanPrune(queryNode, referenceNode))
  {
    ++numPrunes;
    return;
  }

  // If both are leaves, we must evaluate the base case.
  if (queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    // Loop through each of the points in each node.
    for (size_t query = queryNode.Begin(); query < queryNode.End(); ++query)
      for (size_t ref = referenceNode.Begin(); ref < referenceNode.End(); ++ref)
        rule.BaseCase(query, ref);
  }
  else if ((!queryNode.IsLeaf()) && referenceNode.IsLeaf())
  {
    // We have to recurse down the query node.
    if (rule.LeftFirst(referenceNode, queryNode))
    {
      Traverse(*queryNode.Left(), referenceNode);
      Traverse(*queryNode.Right(), referenceNode);
    }
    else
    {
      Traverse(*queryNode.Right(), referenceNode);
      Traverse(*queryNode.Left(), referenceNode);
    }
  }
  else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
  {
    // We have to recurse down the reference node.
    if (rule.LeftFirst(queryNode, referenceNode))
    {
      Traverse(queryNode, *referenceNode.Left());
      Traverse(queryNode, *referenceNode.Right());
    }
    else
    {
      Traverse(queryNode, *referenceNode.Right());
      Traverse(queryNode, *referenceNode.Left());
    }
  }
  else
  {
    // We have to recurse down both.
    if (rule.LeftFirst(*queryNode.Left(), referenceNode))
    {
      Traverse(*queryNode.Left(), *referenceNode.Left());
      Traverse(*queryNode.Left(), *referenceNode.Right());
    }
    else
    {
      Traverse(*queryNode.Left(), *referenceNode.Right());
      Traverse(*queryNode.Left(), *referenceNode.Left());
    }

    // Now recurse to the right query child.
    if (rule.LeftFirst(*queryNode.Right(), referenceNode))
    {
      Traverse(*queryNode.Right(), *referenceNode.Left());
      Traverse(*queryNode.Right(), *referenceNode.Right());
    }
    else
    {
      Traverse(*queryNode.Right(), *referenceNode.Right());
      Traverse(*queryNode.Right(), *referenceNode.Left());
    }
  }

  // Now update any necessary information after recursion.
  rule.UpdateAfterRecursion(queryNode, referenceNode);
}

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

