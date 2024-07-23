/**
 * @file methods/range_search/range_search_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of rules for range search with generic trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_IMPL_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "range_search_rules.hpp"

namespace mlpack {

template<typename DistanceType, typename TreeType>
RangeSearchRules<DistanceType, TreeType>::RangeSearchRules(
    const MatType& referenceSet,
    const MatType& querySet,
    const RangeType<ElemType>& range,
    std::vector<std::vector<size_t> >& neighbors,
    std::vector<std::vector<ElemType> >& distances,
    DistanceType& distance,
    const bool sameSet) :
    referenceSet(referenceSet),
    querySet(querySet),
    range(range),
    neighbors(neighbors),
    distances(distances),
    distance(distance),
    sameSet(sameSet),
    lastQueryIndex(querySet.n_cols),
    lastReferenceIndex(referenceSet.n_cols),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

//! The base case.  Evaluate the distance between the two points and add to the
//! results if necessary.
template<typename DistanceType, typename TreeType>
inline mlpack_force_inline
typename RangeSearchRules<DistanceType, TreeType>::ElemType
RangeSearchRules<DistanceType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If the datasets are the same, don't return the point as in its own range.
  if (sameSet && (queryIndex == referenceIndex))
    return 0.0;

  // If we have just performed this base case, don't do it again.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return 0.0; // No value to return... this shouldn't do anything bad.

  const ElemType d = distance.Evaluate(querySet.unsafe_col(queryIndex),
      referenceSet.unsafe_col(referenceIndex));
  ++baseCases;

  // Update last indices, so we don't accidentally perform a base case twice.
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;

  if (range.Contains(d))
  {
    neighbors[queryIndex].push_back(referenceIndex);
    distances[queryIndex].push_back(d);
  }

  return d;
}

//! Single-tree scoring function.
template<typename DistanceType, typename TreeType>
typename RangeSearchRules<DistanceType, TreeType>::ElemType
RangeSearchRules<DistanceType, TreeType>::Score(const size_t queryIndex,
                                                TreeType& referenceNode)
{
  // We must get the minimum and maximum distances and store them in this
  // object.
  RangeType<ElemType> distances;

  if (TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // In this situation, we calculate the base case.  So we should check to be
    // sure we haven't already done that.
    ElemType baseCase;
    if (TreeTraits<TreeType>::HasSelfChildren &&
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
    ++scores;
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
template<typename DistanceType, typename TreeType>
typename RangeSearchRules<DistanceType, TreeType>::ElemType
RangeSearchRules<DistanceType, TreeType>::Rescore(
    const size_t /* queryIndex */,
    TreeType& /* referenceNode */,
    const ElemType oldScore) const
{
  // If it wasn't pruned before, it isn't pruned now.
  return oldScore;
}

//! Dual-tree scoring function.
template<typename DistanceType, typename TreeType>
typename RangeSearchRules<DistanceType, TreeType>::ElemType
RangeSearchRules<DistanceType, TreeType>::Score(TreeType& queryNode,
                                                TreeType& referenceNode)
{
  RangeType<ElemType> distances;
  if (TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // It is possible that the base case has already been calculated.
    ElemType baseCase = 0.0;
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
    distances = referenceNode.RangeDistance(queryNode);
    ++scores;
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
template<typename DistanceType, typename TreeType>
typename RangeSearchRules<DistanceType, TreeType>::ElemType
RangeSearchRules<DistanceType, TreeType>::Rescore(
    TreeType& /* queryNode */,
    TreeType& /* referenceNode */,
    const ElemType oldScore) const
{
  // If it wasn't pruned before, it isn't pruned now.
  return oldScore;
}

//! Add all the points in the given node to the results for the given query
//! point.
template<typename DistanceType, typename TreeType>
void RangeSearchRules<DistanceType, TreeType>::AddResult(
    const size_t queryIndex, TreeType& referenceNode)
{
  // Some types of trees calculate the base case evaluation before Score() is
  // called, so if the base case has already been calculated, then we must avoid
  // adding that point to the results again.
  size_t baseCaseMod = 0;
  if (TreeTraits<TreeType>::FirstPointIsCentroid &&
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

    const ElemType d = distance.Evaluate(querySet.unsafe_col(queryIndex),
        referenceNode.Dataset().unsafe_col(referenceNode.Descendant(i)));

    neighbors[queryIndex].push_back(referenceNode.Descendant(i));
    distances[queryIndex].push_back(d);
  }
}

} // namespace mlpack

#endif
