/**
 * @file core/tree/octree/single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the single tree traverser for octrees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_OCTREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_OCTREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

namespace mlpack {

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename RuleType>
Octree<DistanceType, StatisticType, MatType>::SingleTreeTraverser<RuleType>::
    SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{
  // Nothing to do.
}

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename RuleType>
void
Octree<DistanceType, StatisticType, MatType>::SingleTreeTraverser<RuleType>::
    Traverse(const size_t queryIndex, Octree& referenceNode)
{
  // If we are a leaf, run the base cases.
  if (referenceNode.NumChildren() == 0)
  {
    const size_t refBegin = referenceNode.Point(0);
    const size_t refEnd = refBegin + referenceNode.NumPoints();
    for (size_t r = refBegin; r < refEnd; ++r)
      rule.BaseCase(queryIndex, r);
  }
  else
  {
    // If it's the root node, just score it.
    if (referenceNode.Parent() == NULL)
    {
      const double rootScore = rule.Score(queryIndex, referenceNode);
      // If root score is DBL_MAX, don't recurse into that node.
      if (rootScore == DBL_MAX)
      {
        ++numPrunes;
        return;
      }
    }

    // Do a prioritized recursion, by scoring all candidates and then sorting
    // them.
    arma::vec scores(referenceNode.NumChildren());
    for (size_t i = 0; i < scores.n_elem; ++i)
      scores[i] = rule.Score(queryIndex, referenceNode.Child(i));

    // Sort the scores.
    arma::uvec sortedIndices = arma::sort_index(scores);

    for (size_t i = 0; i < sortedIndices.n_elem; ++i)
    {
      // If the node is pruned, all subsequent nodes in sorted order will also
      // be pruned.
      if (scores[sortedIndices[i]] == DBL_MAX)
      {
        numPrunes += (sortedIndices.n_elem - i);
        break;
      }

      Traverse(queryIndex, referenceNode.Child(sortedIndices[i]));
    }
  }
}

} // namespace mlpack

#endif
