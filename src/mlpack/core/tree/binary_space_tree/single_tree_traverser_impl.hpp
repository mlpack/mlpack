/**
 * @file single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * A nested class of BinarySpaceTree which traverses the entire tree with a
 * given set of rules which indicate the branches which can be pruned and the
 * order in which to recurse.  This traverser is a depth-first traverser.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

#include <stack>

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
void BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        referenceNode)
{
  // If we are a leaf, run the base case as necessary.
  if (referenceNode.IsLeaf())
  {
    const size_t refEnd = referenceNode.Begin() + referenceNode.Count();
    for (size_t i = referenceNode.Begin(); i < refEnd; ++i)
      rule.BaseCase(queryIndex, i);
  }
  else
  {
    // If either score is DBL_MAX, we do not recurse into that node.
    double leftScore = rule.Score(queryIndex, *referenceNode.Left());
    double rightScore = rule.Score(queryIndex, *referenceNode.Right());

    if (leftScore < rightScore)
    {
      // Recurse to the left.
      Traverse(queryIndex, *referenceNode.Left());

      // Is it still valid to recurse to the right?
      rightScore = rule.Rescore(queryIndex, *referenceNode.Right(), rightScore);

      if (rightScore != DBL_MAX)
        Traverse(queryIndex, *referenceNode.Right()); // Recurse to the right.
      else
        ++numPrunes;
    }
    else if (rightScore < leftScore)
    {
      // Recurse to the right.
      Traverse(queryIndex, *referenceNode.Right());

      // Is it still valid to recurse to the left?
      leftScore = rule.Rescore(queryIndex, *referenceNode.Left(), leftScore);

      if (leftScore != DBL_MAX)
        Traverse(queryIndex, *referenceNode.Left()); // Recurse to the left.
      else
        ++numPrunes;
    }
    else // leftScore is equal to rightScore.
    {
      if (leftScore == DBL_MAX)
      {
        numPrunes += 2; // Pruned both left and right.
      }
      else
      {
        // Choose the left first.
        Traverse(queryIndex, *referenceNode.Left());

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(queryIndex, *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
          Traverse(queryIndex, *referenceNode.Right());
        else
          ++numPrunes;
      }
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif
