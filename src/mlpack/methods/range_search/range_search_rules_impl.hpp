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
namespace neighbor {

template<typename MetricType, typename TreeType>
RangeSearchRules::RangeSearchRules(const arma::mat& referenceSet,
                                   const arma::mat& querySet,
                                   std::vector<std::vector<size_t> >& neighbors,
                                   std::vector<std::vector<double> >& distances,
                                   math::Range& range,
                                   MetricType& metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    neighbors(neighbors),
    distances(distances),
    range(range),
    metric(metric)
{
  // Nothing to do.
}

//! The base case.  Evaluate the distance between the two points and add to the
//! results if necessary.
template<typename MetricType, typename TreeType>
double RangeSearchRules::BaseCase(const size_t queryIndex,
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
double RangeSearchRules::Score(const size_t queryIndex,
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
    AddResult(queryIndex, referenceNode);
    return DBL_MAX; // We don't need to go any deeper.
  }

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in range
  // search.
  return 0.0;
}

//! Single-tree scoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules::Score(const size_t queryIndex,
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
    AddResult(queryIndex, referenceNode);
    return DBL_MAX; // We don't need to go any deeper.
  }

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in range
  // search.
  return 0.0;
}

//! Single-tree rescoring function.
template<typename MetricType, typename TreeType>
double RangeSearchRules<MetricType, TreeType>::Rescore(const size_t queryIndex,
                                 TreeType& referenceNode,
                                 const double oldScore)
{
  // If it wasn't pruned before, it isn't pruned now.
  return oldScore;
}

}; // namespace neighbor
}; // namespace mlpack

#endif
