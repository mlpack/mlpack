/**
 * @file dual_tree_traverser_impl.hpp
 *
 * Implementation of the DualTreeTraverser for VantagePointTree.  This is a way
 * to perform a dual-tree traversal of two trees.  The trees must be the same
 * type.
 */
#ifndef MLPACK_CORE_TREE_VANTAGE_POINT_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_VANTAGE_POINT_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_traverser.hpp"

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
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do. */ }

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType, size_t...>
             class SplitType>
template<typename RuleType>
void VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
DualTreeTraverser<RuleType>::Traverse(
    VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        queryNode,
    VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        referenceNode)
{
  typedef typename RuleType::TraversalInfoType TravInfo;
  // This tree traverser use TraversalInfoType slightly different than other
  // traversers. All children at one level use the same traversal information
  // since their centroids are equal.

  // Increment the visit counter.
  ++numVisited;

  // If both are leaves, we must evaluate the base case.
  if (queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    TravInfo traversalInfo = rule.TraversalInfo();

    if (traversalInfo.LastQueryNode() == &queryNode &&
        traversalInfo.LastReferenceNode() == &referenceNode)
        return; // We have already calculated this base case.

    // Loop through each of the points in each node.
    const size_t queryEnd = queryNode.Begin() + queryNode.Count();
    const size_t refEnd = referenceNode.Begin() + referenceNode.Count();
    for (size_t query = queryNode.Begin(); query < queryEnd; ++query)
    {
      // See if we need to investigate this point (this function should be
      // implemented for the single-tree recursion too).
      const double childScore = rule.Score(query, referenceNode);

      if (childScore == DBL_MAX)
        continue; // We can't improve this particular point.

      for (size_t ref = referenceNode.Begin(); ref < refEnd; ++ref)
        rule.BaseCase(query, ref);

      numBaseCases += referenceNode.Count();
    }
  }
  else if (((!queryNode.IsLeaf()) && referenceNode.IsLeaf()) ||
           (queryNode.NumDescendants() > 3 * referenceNode.NumDescendants() &&
            !queryNode.IsLeaf() && !referenceNode.IsLeaf()))
  {
    // We have to recurse down the query node.  In this case the recursion order
    // does not matter.

    const double pointScore = rule.Score(*queryNode.Central(), referenceNode);
    ++numScores;

    // The traversal information is the same for all children.
    TravInfo traversalInfo = rule.TraversalInfo();

    if (pointScore != DBL_MAX)
      Traverse(*queryNode.Central(), referenceNode);
    else
      ++numPrunes;

    // Before recursing, we have to set the traversal information correctly.
    rule.TraversalInfo() = traversalInfo;

    const double innerScore = rule.Score(*queryNode.Inner(), referenceNode);
    ++numScores;

    if (innerScore != DBL_MAX)
      Traverse(*queryNode.Inner(), referenceNode);
    else
      ++numPrunes;

    // Before recursing, we have to set the traversal information correctly.
    rule.TraversalInfo() = traversalInfo;
    const double outerScore = rule.Score(*queryNode.Outer(), referenceNode);
    ++numScores;

    if (outerScore != DBL_MAX)
      Traverse(*queryNode.Outer(), referenceNode);
    else
      ++numPrunes;
  }
  else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
  {
    // We have to recurse down the reference node.  In this case the recursion
    // order does matter.
    TraverseReferenceNode(queryNode, referenceNode);
  }
  else
  {
    // We have to recurse down both query and reference nodes.  Because the
    // query descent order does not matter, we will go to the central query
    // child first.  Before recursing, we have to set the traversal information
    // correctly.

    typename RuleType::TraversalInfoType traversalInfo;
 
    // We have to calculate the base case with the central reference node.
    // All children of a vantage point tree node use the same traversal
    // information.
    traversalInfo.LastQueryNode() = queryNode.Central();
    traversalInfo.LastReferenceNode() = referenceNode.Central();
    traversalInfo.LastBaseCase() = rule.BaseCase(queryNode.Central()->Point(0),
        referenceNode.Central()->Point(0));
    traversalInfo.LastScore() = traversalInfo.LastBaseCase();
    numBaseCases++;

    rule.TraversalInfo() = traversalInfo;

    // Now recurse down the central node.
    TraverseReferenceNode(*queryNode.Central(), referenceNode);

    // Restore the main traversal information.
    rule.TraversalInfo() = traversalInfo;

    // Now recurse down the inner node.
    TraverseReferenceNode(*queryNode.Inner(), referenceNode);

    // Restore the main traversal information.
    rule.TraversalInfo() = traversalInfo;

    // Now recurse down the outer query node.
    TraverseReferenceNode(*queryNode.Outer(), referenceNode);
  }
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType, size_t...>
             class SplitType>
template<typename RuleType>
void VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
DualTreeTraverser<RuleType>::TraverseReferenceNode(
    VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        queryNode,
    VantagePointTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        referenceNode)
{
  typedef VantagePointTree<MetricType, StatisticType, MatType, BoundType,
      SplitType> TreeType;
  typedef typename RuleType::TraversalInfoType TravInfo;

  // We have to recurse down the reference node.  In this case the recursion
  // order does matter.  Before recursing, though, we have to set the
  // traversal information correctly.

  std::array<std::tuple<double, TreeType*>, 3> scores;

  double score = rule.Score(queryNode, *referenceNode.Central());
  scores[0] = std::make_tuple(score, referenceNode.Central());

  // All children of a vantage point tree use the same traversal info.
  TravInfo traversalInfo = rule.TraversalInfo();

  score = rule.Score(queryNode, *referenceNode.Inner());
  scores[1] = std::make_tuple(score, referenceNode.Inner());

  score = rule.Score(queryNode, *referenceNode.Outer());
  scores[2] = std::make_tuple(score, referenceNode.Outer());
  numScores += 3;

  // Sort the array according to the score.
  if (std::get<0>(scores[0]) > std::get<0>(scores[1]))
    std::swap(scores[0], scores[1]);
  if (std::get<0>(scores[1]) > std::get<0>(scores[2]))
    std::swap(scores[1], scores[2]);
  if (std::get<0>(scores[0]) > std::get<0>(scores[1]))
    std::swap(scores[0], scores[1]);

  for (size_t i = 0; i < 3; i++)
  {
    if (std::get<0>(scores[i]) == DBL_MAX)
    {
      numPrunes += 3 - i;
      break;
    }

    // Is it still valid to recurse to the node?
    double rescore = 0;
    if (i > 0)
      rescore = rule.Rescore(queryNode, *std::get<1>(scores[i]),
          std::get<0>(scores[i]));

    if (rescore != DBL_MAX)
    {
      // Restore the traversal info.
      rule.TraversalInfo() = traversalInfo;
      Traverse(queryNode, *std::get<1>(scores[i]));
    }
    else
      numPrunes++;
  }
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

