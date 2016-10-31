/**
 * @file spill_single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * A nested class of SpillTree which traverses the entire tree with a
 * given set of rules which indicate the branches which can be pruned and the
 * order in which to recurse.  This traverser is a depth-first traverser.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPILL_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPILL_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "spill_single_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
template<typename RuleType, bool Defeatist>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillSingleTreeTraverser<RuleType, Defeatist>::SpillSingleTreeTraverser(
    RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
template<typename RuleType, bool Defeatist>
void SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillSingleTreeTraverser<RuleType, Defeatist>::Traverse(
    const size_t queryIndex,
    SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>&
        referenceNode)
{
  // If we are a leaf, run the base case as necessary.
  if (referenceNode.IsLeaf())
  {
    for (size_t i = 0; i < referenceNode.NumPoints(); ++i)
      rule.BaseCase(queryIndex, referenceNode.Point(i));
  }
  else
  {
    if (Defeatist && referenceNode.Overlap())
    {
      // If referenceNode is a overlapping node we do defeatist search.
      size_t bestChild = rule.GetBestChild(queryIndex, referenceNode);
      Traverse(queryIndex, referenceNode.Child(bestChild));
      ++numPrunes;
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
        rightScore = rule.Rescore(queryIndex, *referenceNode.Right(),
            rightScore);

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
}

} // namespace tree
} // namespace mlpack

#endif
