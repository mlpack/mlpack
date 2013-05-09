/**
 * @file range_search_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of rules for range search with generic trees.
 */
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_IMPL_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "range_search_rules.hpp"

namespace mlpack {
namespace range {

template<typename MetricType, typename TreeType>
RangeSearchRules<MetricType, TreeType>::RangeSearchRules(
    const arma::mat& referenceSet,
    const arma::mat& querySet,
    const math::Range& range,
    std::vector<std::vector<size_t> >& neighbors,
    std::vector<std::vector<double> >& distances,
    MetricType& metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    range(range),
    neighbors(neighbors),
    distances(distances),
    metric(metric)
{
  // Nothing to do.
}

//! The base case.  Evaluate the distance between the two points and add to the
//! results if necessary.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If the datasets are the same, don't return the point as in its own range.
  if ((&referenceSet == &querySet) && (queryIndex == referenceIndex))
    return 0.0;

  const double distance = metric.Evaluate(querySet.unsafe_col(queryIndex),
      referenceSet.unsafe_col(referenceIndex));

  if (range.Contains(distance))
  {
    neighbors[queryIndex].push_back(referenceIndex);
    distances[queryIndex].push_back(distance);
  }

  return distance;
}

//! Single-tree scoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::Score(const size_t queryIndex,
                                                     TreeType& referenceNode)
{
  const math::Range distances =
      referenceNode.RangeDistance(querySet.unsafe_col(queryIndex));

  // If the ranges do not overlap, prune this node.
  if (!distances.Contains(range))
    return DBL_MAX;

  // In this case, all of the points in the reference node will be part of the
  // results.
  if ((distances.Lo() >= range.Lo()) && (distances.Hi() <= range.Hi()))
  {
    AddResult(queryIndex, referenceNode, false);
    return DBL_MAX; // We don't need to go any deeper.
  }

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in range
  // search.
  return 0.0;
}

//! Single-tree scoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double baseCaseResult)
{
  const math::Range distances = referenceNode.RangeDistance(
      querySet.unsafe_col(queryIndex), baseCaseResult);

  // If the ranges do not overlap, prune this node.
  if (!distances.Contains(range))
    return DBL_MAX;

  // In this case, all of the points in the reference node will be part of the
  // results.
  if ((distances.Lo() >= range.Lo()) && (distances.Hi() <= range.Hi()))
  {
    AddResult(queryIndex, referenceNode, true);
    return DBL_MAX; // We don't need to go any deeper.
  }

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in range
  // search.
  return 0.0;
}

//! Single-tree rescoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::Rescore(
    const size_t /* queryIndex */,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  // If it wasn't pruned before, it isn't pruned now.
  return oldScore;
}

//! Dual-tree scoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::Score(TreeType& queryNode,
                                                     TreeType& referenceNode)
{
  const math::Range distances = referenceNode.RangeDistance(&queryNode);

  // If the ranges do not overlap, prune this node.
  if (!distances.Contains(range))
    return DBL_MAX;

  // In this case, all of the points in the reference node will be part of all
  // the results for each point in the query node.
  if ((distances.Lo() >= range.Lo()) && (distances.Hi() <= range.Hi()))
  {
    for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
      AddResult(queryNode.Descendant(i), referenceNode, false);
    return DBL_MAX; // We don't need to go any deeper.
  }

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in range
  // search.
  return 0.0;
}

//! Dual-tree scoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double baseCaseResult)
{
  const math::Range distances = referenceNode.RangeDistance(&queryNode,
      baseCaseResult);

  // If the ranges do not overlap, prune this node.
  if (!distances.Contains(range))
    return DBL_MAX;

  // In this case, all of the points in the reference node will be part of all
  // the results for each point in the query node.
  if ((distances.Lo() >= range.Lo()) && (distances.Hi() <= range.Hi()))
  {
    AddResult(queryNode.Descendant(0), referenceNode, true);
    // We have not calculated the base case for any descendants other than the
    // first point.
    for (size_t i = 1; i < queryNode.NumDescendants(); ++i)
      AddResult(queryNode.Descendant(i), referenceNode, false);
    return DBL_MAX; // We don't need to go any deeper.
  }

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in range
  // search.
  return 0.0;
}

//! Dual-tree rescoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::Rescore(
    TreeType& /* queryNode */,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  // If it wasn't pruned before, it isn't pruned now.
  return oldScore;
}

//! Add all the points in the given node to the results for the given query
//! point.
template<typename MetricType, typename TreeType>
void RangeSearchRules<MetricType, TreeType>::AddResult(const size_t queryIndex,
                                                       TreeType& referenceNode,
                                                       const bool hasBaseCase)
{
  // Some types of trees calculate the base case evaluation before Score() is
  // called, so if the base case has already been calculated, then we must avoid
  // adding that point to the results again.
  size_t baseCaseMod = 0;
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid && hasBaseCase)
  {
    baseCaseMod = 1;
  }

  // Resize distances and neighbors vectors appropriately.  We have to use
  // reserve() and not resize(), because we don't know if we will encounter the
  // case where the datasets and points are the same (and we skip in that case).
  const size_t oldSize = neighbors[queryIndex].size();
  neighbors[queryIndex].reserve(oldSize + referenceNode.NumDescendants() -
      baseCaseMod);
  distances[queryIndex].reserve(oldSize + referenceNode.NumDescendants() -
      baseCaseMod);

  for (size_t i = baseCaseMod; i < referenceNode.NumDescendants(); ++i)
  {
    if ((&referenceSet == &querySet) &&
        (queryIndex == referenceNode.Descendant(i)))
      continue;

    const double distance = metric.Evaluate(querySet.unsafe_col(queryIndex),
        referenceNode.Dataset().unsafe_col(referenceNode.Descendant(i)));

    neighbors[queryIndex].push_back(referenceNode.Descendant(i));
    distances[queryIndex].push_back(distance);
  }
}

}; // namespace range
}; // namespace mlpack

#endif
