/**
 * @file nearest_neighbor_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of NearestNeighborRules.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "neighbor_search_rules.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearchRules<SortPolicy, MetricType, TreeType>::NeighborSearchRules(
    const typename TreeType::Mat& referenceSet,
    const typename TreeType::Mat& querySet,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances,
    MetricType& metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    neighbors(neighbors),
    distances(distances),
    metric(metric),
    lastQueryIndex(querySet.n_cols),
    lastReferenceIndex(referenceSet.n_cols),
    baseCases(0),
    scores(0)
{
  // We must set the traversal info last query and reference node pointers to
  // something that is both invalid (i.e. not a tree node) and not NULL.  We'll
  // use the this pointer.
  traversalInfo.LastQueryNode() = (TreeType*) this;
  traversalInfo.LastReferenceNode() = (TreeType*) this;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline force_inline // Absolutely MUST be inline so optimizations can happen.
double NeighborSearchRules<SortPolicy, MetricType, TreeType>::
BaseCase(const size_t queryIndex, const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return 0.0;

  // If we have already performed this base case, then do not perform it again.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return lastBaseCase;

  double distance = metric.Evaluate(querySet.col(queryIndex),
                                    referenceSet.col(referenceIndex));
  ++baseCases;

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distances.unsafe_col(queryIndex);
  arma::Col<size_t> queryIndices = neighbors.unsafe_col(queryIndex);
  const size_t insertPosition = SortPolicy::SortDistance(queryDist,
      queryIndices, distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, distance);

  // Cache this information for the next time BaseCase() is called.
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;
  lastBaseCase = distance;

  return distance;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  ++scores; // Count number of Score() calls.
  double distance;
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // The first point in the tree is the centroid.  So we can then calculate
    // the base case between that and the query point.
    double baseCase = -1.0;
    if (tree::TreeTraits<TreeType>::HasSelfChildren)
    {
      // If the parent node is the same, then we have already calculated the
      // base case.
      if ((referenceNode.Parent() != NULL) &&
          (referenceNode.Point(0) == referenceNode.Parent()->Point(0)))
        baseCase = referenceNode.Parent()->Stat().LastDistance();
      else
        baseCase = BaseCase(queryIndex, referenceNode.Point(0));

      // Save this evaluation.
      referenceNode.Stat().LastDistance() = baseCase;
    }

    distance = SortPolicy::CombineBest(baseCase,
        referenceNode.FurthestDescendantDistance());
  }
  else
  {
    distance = SortPolicy::BestPointToNodeDistance(querySet.col(queryIndex),
        &referenceNode);
  }

  // Compare against the best k'th distance for this query point so far.
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Rescore(
    const size_t queryIndex,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  // If we are already pruning, still prune.
  if (oldScore == DBL_MAX)
    return oldScore;

  // Just check the score again against the distances.
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  return (SortPolicy::IsBetter(oldScore, bestDistance)) ? oldScore : DBL_MAX;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  ++scores; // Count number of Score() calls.

  // Update our bound.
  const double bestDistance = CalculateBound(queryNode);

  // Use the traversal info to see if a parent-child or parent-parent prune is
  // possible.  This is a looser bound than we could make, but it might be
  // sufficient.
  const double queryParentDist = queryNode.ParentDistance();
  const double queryDescDist = queryNode.FurthestDescendantDistance();
  const double refParentDist = referenceNode.ParentDistance();
  const double refDescDist = referenceNode.FurthestDescendantDistance();
  const double score = traversalInfo.LastScore();
  double adjustedScore;

  // We want to set adjustedScore to be the distance between the centroid of the
  // last query node and last reference node.  We will do this by adjusting the
  // last score.  In some cases, we can just use the last base case.
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    adjustedScore = traversalInfo.LastBaseCase();
  }
  else if (score == 0.0) // Nothing we can do here.
  {
    adjustedScore = 0.0;
  }
  else
  {
    // The last score is equal to the distance between the centroids minus the
    // radii of the query and reference bounds along the axis of the line
    // between the two centroids.  In the best case, these radii are the
    // furthest descendant distances, but that is not always true.  It would
    // take too long to calculate the exact radii, so we are forced to use
    // MinimumBoundDistance() as a lower-bound approximation.
    const double lastQueryDescDist =
        traversalInfo.LastQueryNode()->MinimumBoundDistance();
    const double lastRefDescDist =
        traversalInfo.LastReferenceNode()->MinimumBoundDistance();
    adjustedScore = SortPolicy::CombineWorst(score, lastQueryDescDist);
    adjustedScore = SortPolicy::CombineWorst(score, lastRefDescDist);
  }

  // Assemble an adjusted score.  For nearest neighbor search, this adjusted
  // score is a lower bound on MinDistance(queryNode, referenceNode) that is
  // assembled without actually calculating MinDistance().  For furthest
  // neighbor search, it is an upper bound on
  // MaxDistance(queryNode, referenceNode).  If the traversalInfo isn't usable
  // then the node should not be pruned by this.
  if (traversalInfo.LastQueryNode() == queryNode.Parent())
  {
    const double queryAdjust = queryParentDist + queryDescDist;
    adjustedScore = SortPolicy::CombineBest(adjustedScore, queryAdjust);
  }
  else if (traversalInfo.LastQueryNode() == &queryNode)
  {
    adjustedScore = SortPolicy::CombineBest(adjustedScore, queryDescDist);
  }
  else
  {
    // The last query node wasn't this query node or its parent.  So we force
    // the adjustedScore to be such that this combination can't be pruned here,
    // because we don't really know anything about it.

    // It would be possible to modify this section to try and make a prune based
    // on the query descendant distance and the distance between the query node
    // and last traversal query node, but this case doesn't actually happen for
    // kd-trees or cover trees.
    adjustedScore = SortPolicy::BestDistance();
  }

  if (traversalInfo.LastReferenceNode() == referenceNode.Parent())
  {
    const double refAdjust = refParentDist + refDescDist;
    adjustedScore = SortPolicy::CombineBest(adjustedScore, refAdjust);
  }
  else if (traversalInfo.LastReferenceNode() == &referenceNode)
  {
    adjustedScore = SortPolicy::CombineBest(adjustedScore, refDescDist);
  }
  else
  {
    // The last reference node wasn't this reference node or its parent.  So we
    // force the adjustedScore to be such that this combination can't be pruned
    // here, because we don't really know anything about it.

    // It would be possible to modify this section to try and make a prune based
    // on the reference descendant distance and the distance between the
    // reference node and last traversal reference node, but this case doesn't
    // actually happen for kd-trees or cover trees.
    adjustedScore = SortPolicy::BestDistance();
  }

  // Can we prune?
  if (SortPolicy::IsBetter(bestDistance, adjustedScore))
  {
    if (!(tree::TreeTraits<TreeType>::FirstPointIsCentroid && score == 0.0))
    {
      // There isn't any need to set the traversal information because no
      // descendant combinations will be visited, and those are the only
      // combinations that would depend on the traversal information.
      return DBL_MAX;
    }
  }

  double distance;
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    // The first point in the node is the centroid, so we can calculate the
    // distance between the two points using BaseCase() and then find the
    // bounds.  This is potentially loose for non-ball bounds.
    double baseCase = -1.0;
    if (tree::TreeTraits<TreeType>::HasSelfChildren &&
       (traversalInfo.LastQueryNode()->Point(0) == queryNode.Point(0)) &&
       (traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0)))
    {
      // We already calculated it.
      baseCase = traversalInfo.LastBaseCase();
    }
    else
    {
      baseCase = BaseCase(queryNode.Point(0), referenceNode.Point(0));
    }

    distance = SortPolicy::CombineBest(baseCase,
        queryNode.FurthestDescendantDistance() +
        referenceNode.FurthestDescendantDistance());

    lastQueryIndex = queryNode.Point(0);
    lastReferenceIndex = referenceNode.Point(0);
    lastBaseCase = baseCase;

    traversalInfo.LastBaseCase() = baseCase;
  }
  else
  {
    distance = SortPolicy::BestNodeToNodeDistance(&queryNode, &referenceNode);
  }

  if (SortPolicy::IsBetter(distance, bestDistance))
  {
    // Set traversal information.
    traversalInfo.LastQueryNode() = &queryNode;
    traversalInfo.LastReferenceNode() = &referenceNode;
    traversalInfo.LastScore() = distance;

    return distance;
  }
  else
  {
    // There isn't any need to set the traversal information because no
    // descendant combinations will be visited, and those are the only
    // combinations that would depend on the traversal information.
    return DBL_MAX;
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Rescore(
    TreeType& queryNode,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  if (oldScore == DBL_MAX)
    return oldScore;

  // Update our bound.
  const double bestDistance = CalculateBound(queryNode);

  return (SortPolicy::IsBetter(oldScore, bestDistance)) ? oldScore : DBL_MAX;
}

// Calculate the bound for a given query node in its current state and update
// it.
template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::
    CalculateBound(TreeType& queryNode) const
{
  // We have five possible bounds, and we must take the best of them all.  We
  // don't use min/max here, but instead "best/worst", because this is general
  // to the nearest-neighbors/furthest-neighbors cases.  For nearest neighbors,
  // min = best, max = worst.
  //
  // (1) worst ( worst_{all points p in queryNode} D_p[k],
  //             worst_{all children c in queryNode} B(c) );
  // (2) best_{all points p in queryNode} D_p[k] + worst child distance +
  //        worst descendant distance;
  // (3) best_{all children c in queryNode} B(c) +
  //      2 ( worst descendant distance of queryNode -
  //          worst descendant distance of c );
  // (4) B_1(parent of queryNode)
  // (5) B_2(parent of queryNode);
  //
  // D_p[k] is the current k'th candidate distance for point p.
  // So we will loop over the points in queryNode and the children in queryNode
  // to calculate all five of these quantities.

  // Hm, can we populate our distances vector with estimates from the parent?
  // This is written specifically for the cover tree and assumes only one point
  // in a node.
//  if (queryNode.Parent() != NULL && queryNode.NumPoints() > 0)
//  {
//    size_t parentIndexStart = 0;
//    for (size_t i = 0; i < neighbors.n_rows; ++i)
//    {
//      const double pointDistance = distances(i, queryNode.Point(0));
//      if (pointDistance == DBL_MAX)
//      {
//      // Cool, can we take an estimate from the parent?
//        const double parentWorstBound = distances(distances.n_rows - 1,
//              queryNode.Parent()->Point(0));
//        if (parentWorstBound != DBL_MAX)
//        {
//          const double parentAdjustedDistance = parentWorstBound +
//              queryNode.ParentDistance();
//          distances(i, queryNode.Point(0)) = parentAdjustedDistance;
//        }
//      }
//    }
//  }

  double worstPointDistance = SortPolicy::BestDistance();
  double bestPointDistance = SortPolicy::WorstDistance();

  // Loop over all points in this node to find the best and worst distance
  // candidates (for (1) and (2)).
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double distance = distances(distances.n_rows - 1,
        queryNode.Point(i));
    if (SortPolicy::IsBetter(distance, bestPointDistance))
      bestPointDistance = distance;
    if (SortPolicy::IsBetter(worstPointDistance, distance))
      worstPointDistance = distance;
  }

  // Loop over all the children in this node to find the worst bound (for (1))
  // and the best bound with the correcting factor for descendant distances (for
  // (3)).
  double worstChildBound = SortPolicy::BestDistance();
  double bestAdjustedChildBound = SortPolicy::WorstDistance();
  const double queryMaxDescendantDistance =
      queryNode.FurthestDescendantDistance();

  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double firstBound = queryNode.Child(i).Stat().FirstBound();
    const double secondBound = queryNode.Child(i).Stat().SecondBound();
    const double childMaxDescendantDistance =
        queryNode.Child(i).FurthestDescendantDistance();

    if (SortPolicy::IsBetter(worstChildBound, firstBound))
      worstChildBound = firstBound;

    // Now calculate adjustment for maximum descendant distances.
    const double adjustedBound = SortPolicy::CombineWorst(secondBound,
        2 * (queryMaxDescendantDistance - childMaxDescendantDistance));
    if (SortPolicy::IsBetter(adjustedBound, bestAdjustedChildBound))
      bestAdjustedChildBound = adjustedBound;
  }

  // This is bound (1).
  const double firstBound =
      (SortPolicy::IsBetter(worstPointDistance, worstChildBound)) ?
      worstChildBound : worstPointDistance;

  // This is bound (2).
  const double secondBound = SortPolicy::CombineWorst(
      SortPolicy::CombineWorst(bestPointDistance, queryMaxDescendantDistance),
      queryNode.FurthestPointDistance());

  // Bound (3) is bestAdjustedChildBound.

  // Bounds (4) and (5) are the parent bounds.
  const double fourthBound = (queryNode.Parent() != NULL) ?
      queryNode.Parent()->Stat().FirstBound() : SortPolicy::WorstDistance();
//  const double fifthBound = (queryNode.Parent() != NULL) ?
//      queryNode.Parent()->Stat().SecondBound() -
//      queryNode.Parent()->FurthestDescendantDistance() -
//      queryNode.Parent()->FurthestPointDistance() + queryMaxDescendantDistance +
//      queryNode.FurthestPointDistance() + queryNode.ParentDistance() :
//      SortPolicy::WorstDistance();

  // Now, we will take the best of these.  Unfortunately due to the way
  // IsBetter() is defined, this sort of has to be a little ugly.
  // The variable interA represents the first bound (B_1), which is the worst
  // candidate distance of any descendants of this node.
  // The variable interC represents the second bound (B_2), which is a bound on
  // the worst distance of any descendants of this node assembled using the best
  // descendant candidate distance modified using the furthest descendant
  // distance.
  const double interA = (SortPolicy::IsBetter(firstBound, fourthBound)) ?
      firstBound : fourthBound;
  const double interB =
      (SortPolicy::IsBetter(bestAdjustedChildBound, secondBound)) ?
      bestAdjustedChildBound : secondBound;
//  const double interC = (SortPolicy::IsBetter(interB, fifthBound)) ? interB :
//      fifthBound;

  // Update the first and second bounds of the node.
  queryNode.Stat().FirstBound() = interA;
  queryNode.Stat().SecondBound() = interB;

  // Update the actual bound of the node.
  queryNode.Stat().Bound() = (SortPolicy::IsBetter(interA, interB)) ? interB :
      interB;

  return queryNode.Stat().Bound();
}

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearchRules<SortPolicy, MetricType, TreeType>::InsertNeighbor(
    const size_t queryIndex,
    const size_t pos,
    const size_t neighbor,
    const double distance)
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (distances.n_rows - 1))
  {
    int len = (distances.n_rows - 1) - pos;
    memmove(distances.colptr(queryIndex) + (pos + 1),
        distances.colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(neighbors.colptr(queryIndex) + (pos + 1),
        neighbors.colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  distances(pos, queryIndex) = distance;
  neighbors(pos, queryIndex) = neighbor;
}

}; // namespace neighbor
}; // namespace mlpack

#endif // __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP
