/**
 * @file single_tree_traverser_impl.hpp
 * @author Andrew Wells
 *
 * A class for traversing rectangle type trees with a given set of rules
 * which indicate the branches to prune and the order in which to recurse.
 * This is a depth-first traverser.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

#include "single_tree_traverser.hpp"

#include <algorithm>
#include <stack>

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename RuleType>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do */ }

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename RuleType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    const RectangleTree& referenceNode)
{

  // If we reach a leaf node, we need to run the base case.
  if (referenceNode.IsLeaf())
  {
    for (size_t i = 0; i < referenceNode.Count(); i++)
      rule.BaseCase(queryIndex, referenceNode.Point(i));

    return;
  }

  // This is not a leaf node so we sort the children of this node by their
  // scores.
  std::vector<NodeAndScore> nodesAndScores(referenceNode.NumChildren());
  for (size_t i = 0; i < referenceNode.NumChildren(); i++)
  {
    nodesAndScores[i].node = &(referenceNode.Child(i));
    nodesAndScores[i].score = rule.Score(queryIndex, *nodesAndScores[i].node);
  }

  std::sort(nodesAndScores.begin(), nodesAndScores.end(), NodeComparator);

  // Now iterate through them starting with the best and stopping when we reach
  // one that isn't good enough.
  for (size_t i = 0; i < referenceNode.NumChildren(); i++)
  {
    if (rule.Rescore(queryIndex, *nodesAndScores[i].node,
        nodesAndScores[i].score) != DBL_MAX)
    {
      Traverse(queryIndex, *nodesAndScores[i].node);
    }
    else
    {
      numPrunes += referenceNode.NumChildren() - i;
      return;
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif
