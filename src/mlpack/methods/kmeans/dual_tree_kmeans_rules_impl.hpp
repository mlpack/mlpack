/**
 * @file dtnn_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of DualTreeKMeansRules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_IMPL_HPP
#define MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_IMPL_HPP

#include "dual_tree_kmeans_rules.hpp"

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename TreeType>
DualTreeKMeansRules<MetricType, TreeType>::DualTreeKMeansRules(
    const arma::mat& centroids,
    const arma::mat& dataset,
    arma::Row<size_t>& assignments,
    arma::vec& upperBounds,
    arma::vec& lowerBounds,
    MetricType& metric,
    const std::vector<bool>& prunedPoints,
    const std::vector<size_t>& oldFromNewCentroids,
    std::vector<bool>& visited) :
    centroids(centroids),
    dataset(dataset),
    assignments(assignments),
    upperBounds(upperBounds),
    lowerBounds(lowerBounds),
    metric(metric),
    prunedPoints(prunedPoints),
    oldFromNewCentroids(oldFromNewCentroids),
    visited(visited),
    baseCases(0),
    scores(0),
    lastQueryIndex(dataset.n_cols),
    lastReferenceIndex(centroids.n_cols)
{
  // We must set the traversal info last query and reference node pointers to
  // something that is both invalid (i.e. not a tree node) and not NULL.  We'll
  // use the this pointer.
  traversalInfo.LastQueryNode() = (TreeType*) this;
  traversalInfo.LastReferenceNode() = (TreeType*) this;
}

template<typename MetricType, typename TreeType>
inline force_inline double DualTreeKMeansRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  if (prunedPoints[queryIndex])
    return 0.0; // Returning 0 shouldn't be a problem.

  // If we have already performed this base case, then do not perform it again.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return lastBaseCase;

  // Any base cases imply that we will get a result.
  visited[queryIndex] = true;

  // Calculate the distance.
  ++baseCases;
  const double distance = metric.Evaluate(dataset.col(queryIndex),
                                          centroids.col(referenceIndex));

  if (distance < upperBounds[queryIndex])
  {
    lowerBounds[queryIndex] = upperBounds[queryIndex];
    upperBounds[queryIndex] = distance;
    assignments[queryIndex] = (tree::TreeTraits<TreeType>::RearrangesDataset) ?
        oldFromNewCentroids[referenceIndex] : referenceIndex;
  }
  else if (distance < lowerBounds[queryIndex])
  {
    lowerBounds[queryIndex] = distance;
  }

  // Cache this information for the next time BaseCase() is called.
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;
  lastBaseCase = distance;

  return distance;
}

template<typename MetricType, typename TreeType>
inline double DualTreeKMeansRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& /* referenceNode */)
{
  // If the query point has already been pruned, then don't recurse further.
  if (prunedPoints[queryIndex])
    return DBL_MAX;

  // No pruning at this level; we're not likely to encounter a single query
  // point with a reference node..
  return 0;
}

template<typename MetricType, typename TreeType>
inline double DualTreeKMeansRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  if (queryNode.Stat().StaticPruned() == true)
    return DBL_MAX;

  // Pruned() for the root node must never be set to size_t(-1).
  if (queryNode.Stat().Pruned() == size_t(-1))
  {
    queryNode.Stat().Pruned() = queryNode.Parent()->Stat().Pruned();
    queryNode.Stat().LowerBound() = queryNode.Parent()->Stat().LowerBound();
    queryNode.Stat().Owner() = queryNode.Parent()->Stat().Owner();
  }

  if (queryNode.Stat().Pruned() == centroids.n_cols)
    return DBL_MAX;

  // This looks a lot like the hackery used in NeighborSearchRules to avoid
  // distance computations.  We'll use the traversal info to see if a
  // parent-child or parent-parent prune is possible.
  const double queryParentDist = queryNode.ParentDistance();
  const double queryDescDist = queryNode.FurthestDescendantDistance();
  const double refParentDist = referenceNode.ParentDistance();
  const double refDescDist = referenceNode.FurthestDescendantDistance();
  const double lastScore = traversalInfo.LastScore();
  double adjustedScore;
  double score = 0.0;

  // We want to set adjustedScore to be the distance between the centroid of the
  // last query node and last reference node.  We will do this by adjusting the
  // last score.  In some cases, we can just use the last base case.
  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    adjustedScore = traversalInfo.LastBaseCase();
  }
  else if (lastScore == 0.0) // Nothing we can do here.
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
    adjustedScore = lastScore + lastQueryDescDist;
    adjustedScore = lastScore + lastRefDescDist;
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
    adjustedScore -= queryAdjust;
  }
  else if (traversalInfo.LastQueryNode() == &queryNode)
  {
    adjustedScore -= queryDescDist;
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
    adjustedScore = 0.0;
  }
  if (traversalInfo.LastReferenceNode() == referenceNode.Parent())
  {
    const double refAdjust = refParentDist + refDescDist;
    adjustedScore -= refAdjust;
  }
  else if (traversalInfo.LastReferenceNode() == &referenceNode)
  {
    adjustedScore -= refDescDist;
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
    adjustedScore = 0.0;
  }

  // Now, check if we can prune.
  if (adjustedScore > queryNode.Stat().UpperBound())
  {
    if (!(tree::TreeTraits<TreeType>::FirstPointIsCentroid && score == 0.0))
    {
      // There isn't any need to set the traversal information because no
      // descendant combinations will be visited, and those are the only
      // combinations that would depend on the traversal information.
      if (adjustedScore < queryNode.Stat().LowerBound())
      {
        // If this might affect the lower bound, make it more exact.
        queryNode.Stat().LowerBound() = std::min(queryNode.Stat().LowerBound(),
            queryNode.MinDistance(referenceNode));
        ++scores;
      }

      queryNode.Stat().Pruned() += referenceNode.NumDescendants();
      score = DBL_MAX;
    }
  }

  if (score != DBL_MAX)
  {
    // Get minimum and maximum distances.
    const math::Range distances = queryNode.RangeDistance(referenceNode);

    score = distances.Lo();
    ++scores;
    if (distances.Lo() > queryNode.Stat().UpperBound())
    {
      // The reference node can own no points in this query node.  We may
      // improve the lower bound on pruned nodes, though.
      if (distances.Lo() < queryNode.Stat().LowerBound())
        queryNode.Stat().LowerBound() = distances.Lo();

      // This assumes that reference clusters don't appear elsewhere in the
      // tree.
      queryNode.Stat().Pruned() += referenceNode.NumDescendants();
      score = DBL_MAX;
    }
    else if (distances.Hi() < queryNode.Stat().UpperBound())
    {
      // Tighten upper bound.
      const double tighterBound =
          queryNode.MaxDistance(centroids.col(referenceNode.Descendant(0)));
      ++scores; // Count extra distance calculation.

      if (tighterBound <= queryNode.Stat().UpperBound())
      {
        // We can improve the best estimate.
        queryNode.Stat().UpperBound() = tighterBound;

        // Remember that our upper bound does correspond to a cluster centroid,
        // so it does correspond to a cluster.  We'll mark the cluster as the
        // owner, but note that the node is not truly owned unless
        // Stat().Pruned() is centroids.n_cols.
        queryNode.Stat().Owner() =
            (tree::TreeTraits<TreeType>::RearrangesDataset) ?
            oldFromNewCentroids[referenceNode.Descendant(0)] :
            referenceNode.Descendant(0);
      }
    }
  }

  // Is everything pruned?

  if (queryNode.Stat().Pruned() == centroids.n_cols - 1)
  {
    queryNode.Stat().Pruned() = centroids.n_cols; // Owner() is already set.
    return DBL_MAX;
  }


  // Set traversal information.
  traversalInfo.LastQueryNode() = &queryNode;
  traversalInfo.LastReferenceNode() = &referenceNode;
  traversalInfo.LastScore() = score;

  return score;
}

template<typename MetricType, typename TreeType>
inline double DualTreeKMeansRules<MetricType, TreeType>::Rescore(
    const size_t /* queryIndex */,
    TreeType& /* referenceNode */,
    const double oldScore)
{
  // No rescoring (for now).
  return oldScore;
}

template<typename MetricType, typename TreeType>
inline double DualTreeKMeansRules<MetricType, TreeType>::Rescore(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double oldScore)
{
  if (oldScore == DBL_MAX)
    return DBL_MAX; // It's already pruned.

  // oldScore contains the minimum distance between queryNode and referenceNode.
  // In the time since Score() has been called, the upper bound *may* have
  // tightened.  If it has tightened enough, we may prune this node now.
  if (oldScore > queryNode.Stat().UpperBound())
  {
    // We may still be able to improve the lower bound on pruned nodes.
    if (oldScore < queryNode.Stat().LowerBound())
      queryNode.Stat().LowerBound() = oldScore;

    // This assumes that reference clusters don't appear elsewhere in the tree.
    queryNode.Stat().Pruned() += referenceNode.NumDescendants();
    return DBL_MAX;
  }

  // Also, check if everything has been pruned.
  if (queryNode.Stat().Pruned() == centroids.n_cols - 1)
  {
    queryNode.Stat().Pruned() = centroids.n_cols; // Owner() is already set.
    return DBL_MAX;
  }

  return oldScore;
}

} // namespace kmeans
} // namespace mlpack

#endif
