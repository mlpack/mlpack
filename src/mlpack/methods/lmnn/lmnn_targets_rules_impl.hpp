/**
 * @file lmnn_targets_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of LMNNTargetsRules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_LMNN_TARGETS_RULES_IMPL_HPP
#define MLPACK_METHODS_LMNN_LMNN_TARGETS_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "lmnn_targets_rules.hpp"

namespace mlpack {
namespace lmnn {

template<typename MetricType, typename TreeType>
LMNNTargetsRules<MetricType, TreeType>::
LMNNTargetsRules(
    const typename TreeType::Mat& dataset,
    const size_t k,
    MetricType& metric) :
    dataset(dataset),
    k(k),
    metric(metric),
    lastQueryIndex(dataset.n_cols),
    lastReferenceIndex(dataset.n_cols)
{
  // We must set the traversal info last query and reference node pointers to
  // something that is both invalid (i.e. not a tree node) and not NULL.  We'll
  // use the this pointer.
  traversalInfo.LastQueryNode() = (TreeType*) this;
  traversalInfo.LastReferenceNode() = (TreeType*) this;

  // Let's build the list of candidate neighbors for each query point.  They
  // will be initialized with k candidates: (DBL_MAX, size_t() - 1) The list of
  // candidates will be updated when visiting new points with the BaseCase()
  // method.
  const Candidate def = std::make_pair(DBL_MAX, size_t() - 1);

  std::vector<Candidate> vect(k, def);
  CandidateList pqueue(CandidateCmp(), std::move(vect));

  candidates.reserve(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    candidates.push_back(pqueue);
  }
}

template<typename MetricType, typename TreeType>
void LMNNTargetsRules<MetricType, TreeType>::GetResults(
    const std::vector<size_t>& oldFromNew,
    arma::Mat<size_t>& neighbors,
    arma::mat& neighborDistances)
{
  // We also perform the reverse mapping here.
  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    CandidateList& pqueue = candidates[i];
    const size_t queryIndex = oldFromNew[i];
    for (size_t j = 1; j <= k; ++j)
    {
      neighbors(k - j, queryIndex) = oldFromNew[pqueue.top().second];
      neighborDistances(k - j, queryIndex) = pqueue.top().first;
      pqueue.pop();
    }
  }
};

template<typename MetricType, typename TreeType>
inline force_inline // Absolutely MUST be inline so optimizations can happen.
double LMNNTargetsRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex, const size_t referenceIndex)
{
  // If we have already performed this base case, then do not perform it again.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return lastBaseCase;

  // Don't compare points against themselves.  We always know when we run this
  // that the reference set and query set will be the same, so if that's ever
  // not the case we must change this...
  if (queryIndex == referenceIndex)
    return 0.0;

  double distance = metric.Evaluate(dataset.col(queryIndex),
                                    dataset.col(referenceIndex));

  InsertNeighbor(queryIndex, referenceIndex, distance);

  // Cache this information for the next time BaseCase() is called.
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;
  lastBaseCase = distance;

  return distance;
}

template<typename MetricType, typename TreeType>
inline double LMNNTargetsRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
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

    distance = std::max(baseCase - referenceNode.FurthestDescendantDistance(),
        0.0);
  }
  else
  {
    distance = neighbor::NearestNeighborSort::BestPointToNodeDistance(
        dataset.col(queryIndex), &referenceNode);
  }

  // Compare against the best k'th neighbor and impostor distances for this
  // query point so far.
  double bestDistance = candidates[queryIndex].top().first;

  return (distance <= bestDistance) ? distance : DBL_MAX;
}

template<typename MetricType, typename TreeType>
inline double LMNNTargetsRules<MetricType, TreeType>::Rescore(
    const size_t queryIndex,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  // If we are already pruning, still prune.
  if (oldScore == DBL_MAX)
    return oldScore;

  // Just check the score again against the distances.
  double bestDistance = candidates[queryIndex].top().first;

  return (oldScore <= bestDistance) ? oldScore : DBL_MAX;
}

template<typename MetricType, typename TreeType>
inline double LMNNTargetsRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
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
    adjustedScore = neighbor::NearestNeighborSort::CombineWorst(score,
        lastQueryDescDist);
    adjustedScore = neighbor::NearestNeighborSort::CombineWorst(adjustedScore,
        lastRefDescDist);
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
    adjustedScore = std::max(adjustedScore - queryAdjust, 0.0);
  }
  else if (traversalInfo.LastQueryNode() == &queryNode)
  {
    adjustedScore = std::max(adjustedScore - queryDescDist, 0.0);
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
    adjustedScore = std::max(adjustedScore - refAdjust, 0.0);
  }
  else if (traversalInfo.LastReferenceNode() == &referenceNode)
  {
    adjustedScore = std::max(adjustedScore - refDescDist, 0.0);
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

  // Can we prune?
  if (!(adjustedScore <= bestDistance))
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

    distance = std::max(baseCase - (queryNode.FurthestDescendantDistance() +
        referenceNode.FurthestDescendantDistance()), 0.0);

    lastQueryIndex = queryNode.Point(0);
    lastReferenceIndex = referenceNode.Point(0);
    lastBaseCase = baseCase;

    traversalInfo.LastBaseCase() = baseCase;
  }
  else
  {
    distance = referenceNode.MinDistance(queryNode);
  }

  if (distance <= bestDistance)
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

template<typename MetricType, typename TreeType>
inline double LMNNTargetsRules<MetricType, TreeType>::Rescore(
    TreeType& queryNode,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  if (oldScore == DBL_MAX || oldScore == 0.0)
    return oldScore;

  // Update our bound.
  const double bestDistance = CalculateBound(queryNode);

  return (oldScore <= bestDistance) ? oldScore : DBL_MAX;
}

// Calculate the bound for a given query node in its current state and update
// it.
template<typename MetricType, typename TreeType>
inline double LMNNTargetsRules<MetricType, TreeType>::
CalculateBound(TreeType& queryNode) const
{
  // This is an adapted form of the B_1(N_q) function in the paper
  // ``Tree-Independent Dual-Tree Algorithms'' by Curtin et. al.; the goal is to
  // place a bound on the worst possible distance a point combination could have
  // to improve any of the current neighbor estimates.  If the best possible
  // distance between two nodes is greater than this bound, then the node
  // combination can be pruned (see Score()).

  // This particular form will work best only for depth-first recursions, as it
  // does not consider the B_2(N_q) term from that paper.  This is because it
  // turns out to be complex to adapt that term: we would need to hold one
  // B_2(N_q) for each possible class to properly apply the bound.  Given that
  // extra bookkeeping and complexity, we have chosen here to just use B_1(N_q).
  // Therefore, the cover tree will not be a good choice for these rules.

  // In this particular case, we have to consider both the k'th best impostor
  // and neighbor candidate distances, not just the impostor distances as in
  // LMNNImpostorsRules.

  double worstDistance = 0.0;

  // Loop over points held in the node.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double distance = candidates[queryNode.Point(i)].top().first;
    if (worstDistance < distance)
      worstDistance = distance;
  }

  // Loop over children of the node, and use their cached information to
  // assemble bounds.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (worstDistance < bound)
      worstDistance = bound;
  }

  // Now consider the parent bounds.
  if (queryNode.Parent() != NULL)
  {
    // The parent's bound implies that the bound for this node must be at least
    // as good.  Thus, if the parent bound is better, then take it.
    if (queryNode.Parent()->Stat().Bound() <= worstDistance)
      worstDistance = queryNode.Parent()->Stat().Bound();
  }

  // Could the existing bounds be better?
  if (queryNode.Stat().Bound() <= worstDistance)
    worstDistance = queryNode.Stat().Bound();

  // Cache bounds for later.
  queryNode.Stat().Bound() = worstDistance;

  return worstDistance;
}

/**
 * Helper function to insert a point into the list of candidate neighbors.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename MetricType, typename TreeType>
inline void LMNNTargetsRules<MetricType, TreeType>::InsertNeighbor(
    const size_t queryIndex,
    const size_t neighbor,
    const double distance)
{
  CandidateList& pqueue = candidates[queryIndex];
  Candidate c = std::make_pair(distance, neighbor);

  if (CandidateCmp()(c, pqueue.top()))
  {
    pqueue.pop();
    pqueue.push(c);
  }
}

} // namespace lmnn
} // namespace mlpack

#endif // MLPACK_METHODS_LMNN_LMNN_TARGETS_RULES_IMPL_HPP
