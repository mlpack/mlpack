/**
 * @file core/tree/greedy_single_tree_traverser_impl.hpp
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

  size_t bestChild = rule.GetBestChild(queryIndex, referenceNode);
  size_t numDescendants;

  // Check that referencenode is not a leaf node while calculating number of
  // descendants of it's best child.
  if (!referenceNode.IsLeaf())
    numDescendants = referenceNode.Child(bestChild).NumDescendants();
  else
    numDescendants = referenceNode.NumPoints();

  // If number of descendants are more than rule.MinimumBaseCases() than we can
  // go along with best child otherwise we need to traverse for each descendant
  // to ensure that we calculate at least rule.MinimumBaseCases() number of base
  // cases.
  if (!referenceNode.IsLeaf())
  {
    if (numDescendants > rule.MinimumBaseCases())
    {
      // We are pruning all but one child.
      numPrunes += referenceNode.NumChildren() - 1;
      // Recurse the best child.
      Traverse(queryIndex, referenceNode.Child(bestChild));
    }
    else
    {
      // Run the base case over first minBaseCases number of descendants.
      for (size_t i = 0; i <= rule.MinimumBaseCases(); ++i)
        rule.BaseCase(queryIndex, referenceNode.Descendant(i));
    }
  }
}

} // namespace mlpack

#endif
