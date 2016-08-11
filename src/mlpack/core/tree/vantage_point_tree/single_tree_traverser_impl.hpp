/**
 * @file single_tree_traverser_impl.hpp
 *
 * A nested class of VantagePointTree which traverses the entire tree with a
 * given set of rules which indicate the branches which can be pruned and the
 * order in which to recurse.  This traverser is a depth-first traverser.
 */
#ifndef MLPACK_CORE_TREE_VANTAGE_POINT_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_VANTAGE_POINT_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

#include <stack>

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType, size_t...>
             class SplitType>
template<typename RuleType>
VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType, size_t...>
             class SplitType>
template<typename RuleType>
void VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        referenceNode)
{
  // If we are a leaf, run the base case as necessary.
  if (referenceNode.IsLeaf())
  {
    const size_t refEnd = referenceNode.Begin() + referenceNode.Count();
    for (size_t i = referenceNode.Begin(); i < refEnd; ++i)
      rule.BaseCase(queryIndex, i);
    return;
  }

  rule.BaseCase(queryIndex, referenceNode.Central()->Point(0));

  // If either score is DBL_MAX, we do not recurse into that node.
  double innerScore = rule.Score(queryIndex, *referenceNode.Inner());
  double outerScore = rule.Score(queryIndex, *referenceNode.Outer());

  if (innerScore < outerScore)
  {
    // Recurse to the inner node.
    Traverse(queryIndex, *referenceNode.Inner());

    // Is it still valid to recurse to the outer node?
    outerScore = rule.Rescore(queryIndex, *referenceNode.Outer(), outerScore);

    if (outerScore != DBL_MAX)
      Traverse(queryIndex, *referenceNode.Outer()); // Recurse to the outer.
    else
      ++numPrunes;
  }
  else if (outerScore < innerScore)
  {
    // Recurse to the outer node.
    Traverse(queryIndex, *referenceNode.Outer());

    // Is it still valid to recurse to the inner node?
    innerScore = rule.Rescore(queryIndex, *referenceNode.Inner(), innerScore);

    if (innerScore != DBL_MAX)
      Traverse(queryIndex, *referenceNode.Inner()); // Recurse to the inner.
    else
      ++numPrunes;
  }
  else // innerScore is equal to outerScore.
  {
    if (innerScore == DBL_MAX)
    {
      numPrunes += 2; // Pruned both inner and outer nodes.
    }
    else
    {
      // Choose the inner node first.
      Traverse(queryIndex, *referenceNode.Inner());

      // Is it still valid to recurse to the outer node?
      outerScore = rule.Rescore(queryIndex, *referenceNode.Outer(),
          outerScore);

      if (outerScore != DBL_MAX)
        Traverse(queryIndex, *referenceNode.Outer());
      else
        ++numPrunes;
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

