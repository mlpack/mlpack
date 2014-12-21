/**
 * @file range_search_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of rules for range search with generic trees.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
    metric(metric),
    lastQueryIndex(querySet.n_cols),
    lastReferenceIndex(referenceSet.n_cols)
{
  // Nothing to do.
}

//! The base case.  Evaluate the distance between the two points and add to the
//! results if necessary.
template<typename MetricType, typename TreeType>
inline force_inline
double RangeSearchRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If the datasets are the same, don't return the point as in its own range.
  if ((&referenceSet == &querySet) && (queryIndex == referenceIndex))
    return 0.0;

  // If we have just performed this base case, don't do it again.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return 0.0; // No value to return... this shouldn't do anything bad.

  const double distance = metric.Evaluate(querySet.unsafe_col(queryIndex),
      referenceSet.unsafe_col(referenceIndex));

  // Update last indices, so we don't accidentally perform a base case twice.
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;

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
  // We must get the minimum and maximum distances and store them in this
  // object.
  math::Range distances;

  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // In this situation, we calculate the base case.  So we should check to be
    // sure we haven't already done that.
    double baseCase;
    if (tree::TreeTraits<TreeType>::HasSelfChildren &&
        (referenceNode.Parent() != NULL) &&
        (referenceNode.Point(0) == referenceNode.Parent()->Point(0)))
    {
      // If the tree has self-children and this is a self-child, the base case
      // was already calculated.
      baseCase = referenceNode.Parent()->Stat().LastDistance();
      lastQueryIndex = queryIndex;
      lastReferenceIndex = referenceNode.Point(0);
    }
    else
    {
      // We must calculate the base case by hand.
      baseCase = BaseCase(queryIndex, referenceNode.Point(0));
    }

    // This may be possibly loose for non-ball bound trees.
    distances.Lo() = baseCase - referenceNode.FurthestDescendantDistance();
    distances.Hi() = baseCase + referenceNode.FurthestDescendantDistance();

    // Update last distance calculation.
    referenceNode.Stat().LastDistance() = baseCase;
  }
  else
  {
    distances = referenceNode.RangeDistance(querySet.unsafe_col(queryIndex));
  }

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

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in
  // range search.
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
  math::Range distances;
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // It is possible that the base case has already been calculated.
    double baseCase = 0.0;
    if ((traversalInfo.LastQueryNode() != NULL) &&
        (traversalInfo.LastReferenceNode() != NULL) &&
        (traversalInfo.LastQueryNode()->Point(0) == queryNode.Point(0)) &&
        (traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0)))
    {
      baseCase = traversalInfo.LastBaseCase();

      // Make sure that if BaseCase() is called, we don't duplicate results.
      lastQueryIndex = queryNode.Point(0);
      lastReferenceIndex = referenceNode.Point(0);
    }
    else
    {
      // We must calculate the base case.
      baseCase = BaseCase(queryNode.Point(0), referenceNode.Point(0));
    }

    distances.Lo() = baseCase - queryNode.FurthestDescendantDistance()
        - referenceNode.FurthestDescendantDistance();
    distances.Hi() = baseCase + queryNode.FurthestDescendantDistance()
        + referenceNode.FurthestDescendantDistance();

    // Update the last distances performed for the query and reference node.
    traversalInfo.LastBaseCase() = baseCase;
  }
  else
  {
    // Just perform the calculation.
    distances = referenceNode.RangeDistance(&queryNode);
  }

  // If the ranges do not overlap, prune this node.
  if (!distances.Contains(range))
    return DBL_MAX;

  // In this case, all of the points in the reference node will be part of all
  // the results for each point in the query node.
  if ((distances.Lo() >= range.Lo()) && (distances.Hi() <= range.Hi()))
  {
    for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
      AddResult(queryNode.Descendant(i), referenceNode);
    return DBL_MAX; // We don't need to go any deeper.
  }

  // Otherwise the score doesn't matter.  Recursion order is irrelevant in range
  // search.
  traversalInfo.LastQueryNode() = &queryNode;
  traversalInfo.LastReferenceNode() = &referenceNode;
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
                                                       TreeType& referenceNode)
{
  // Some types of trees calculate the base case evaluation before Score() is
  // called, so if the base case has already been calculated, then we must avoid
  // adding that point to the results again.
  size_t baseCaseMod = 0;
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid &&
      (queryIndex == lastQueryIndex) &&
      (referenceNode.Point(0) == lastReferenceIndex))
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
