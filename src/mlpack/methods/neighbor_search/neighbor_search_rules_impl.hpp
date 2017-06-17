/**
 * @file neighbor_search_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of NeighborSearchRules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "neighbor_search_rules.hpp"
#include <mlpack/core/tree/spill_tree/is_spill_tree.hpp>

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearchRules<SortPolicy, MetricType, TreeType>::NeighborSearchRules(
    const typename TreeType::Mat& referenceSet,
    const typename TreeType::Mat& querySet,
    const size_t k,
    MetricType& metric,
    const double epsilon,
    const bool sameSet) :
    referenceSet(referenceSet),
    querySet(querySet),
    k(k),
    metric(metric),
    sameSet(sameSet),
    epsilon(epsilon),
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

  // Let's build the list of candidate neighbors for each query point.
  // It will be initialized with k candidates: (WorstDistance, size_t() - 1)
  // The list of candidates will be updated when visiting new points with the
  // BaseCase() method.
  const Candidate def = std::make_pair(SortPolicy::WorstDistance(),
      size_t() - 1);

  std::vector<Candidate> vect(k, def);
  CandidateList pqueue(CandidateCmp(), std::move(vect));

  candidates.reserve(querySet.n_cols);
  for (size_t i = 0; i < querySet.n_cols; i++)
    candidates.push_back(pqueue);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearchRules<SortPolicy, MetricType, TreeType>::GetResults(
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  neighbors.set_size(k, querySet.n_cols);
  distances.set_size(k, querySet.n_cols);

  for (size_t i = 0; i < querySet.n_cols; i++)
  {
    CandidateList& pqueue = candidates[i];
    for (size_t j = 1; j <= k; j++)
    {
      neighbors(k - j, i) = pqueue.top().second;
      distances(k - j, i) = pqueue.top().first;
      pqueue.pop();
    }
  }
};

template<typename SortPolicy, typename MetricType, typename TreeType>
inline force_inline // Absolutely MUST be inline so optimizations can happen.
double NeighborSearchRules<SortPolicy, MetricType, TreeType>::
BaseCase(const size_t queryIndex, const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if (sameSet && (queryIndex == referenceIndex))
    return 0.0;

  // If we have already performed this base case, then do not perform it again.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return lastBaseCase;

  double distance = metric.Evaluate(querySet.col(queryIndex),
                                    referenceSet.col(referenceIndex));
  ++baseCases;

  InsertNeighbor(queryIndex, referenceIndex, distance);

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
  double bestDistance = candidates[queryIndex].top().first;
  bestDistance = SortPolicy::Relax(bestDistance, epsilon);

  return (SortPolicy::IsBetter(distance, bestDistance)) ?
      SortPolicy::ConvertToScore(distance) : DBL_MAX;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline size_t NeighborSearchRules<SortPolicy, MetricType, TreeType>::
GetBestChild(const size_t queryIndex, TreeType& referenceNode)
{
  ++scores;
  return SortPolicy::GetBestChild(querySet.col(queryIndex), referenceNode);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline size_t NeighborSearchRules<SortPolicy, MetricType, TreeType>::
GetBestChild(const TreeType& queryNode, TreeType& referenceNode)
{
  ++scores;
  return SortPolicy::GetBestChild(queryNode, referenceNode);
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

  const double distance = SortPolicy::ConvertToDistance(oldScore);

  // Just check the score again against the distances.
  double bestDistance = candidates[queryIndex].top().first;
  bestDistance = SortPolicy::Relax(bestDistance, epsilon);

  return (SortPolicy::IsBetter(distance, bestDistance)) ? oldScore : DBL_MAX;
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
    adjustedScore = SortPolicy::CombineWorst(adjustedScore, lastRefDescDist);
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
  if (!SortPolicy::IsBetter(adjustedScore, bestDistance))
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

    return SortPolicy::ConvertToScore(distance);
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
  if (oldScore == DBL_MAX || oldScore == 0.0)
    return oldScore;

  const double distance = SortPolicy::ConvertToDistance(oldScore);

  // Update our bound.
  const double bestDistance = CalculateBound(queryNode);

  return (SortPolicy::IsBetter(distance, bestDistance)) ? oldScore : DBL_MAX;
}

// Calculate the bound for a given query node in its current state and update
// it.
template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::
    CalculateBound(TreeType& queryNode) const
{
  // This is an adapted form of the B(N_q) function in the paper
  // ``Tree-Independent Dual-Tree Algorithms'' by Curtin et. al.; the goal is to
  // place a bound on the worst possible distance a point combination could have
  // to improve any of the current neighbor estimates.  If the best possible
  // distance between two nodes is greater than this bound, then the node
  // combination can be pruned (see Score()).

  // There are a couple ways we can assemble a bound.  For simplicity, this is
  // described for nearest neighbor search (SortPolicy = NearestNeighborSort),
  // but the code that is written is adapted for whichever SortPolicy.

  // First, we can consider the current worst neighbor candidate distance of any
  // descendant point.  This is assembled with 'worstDistance' by looping
  // through the points held by the query node, and then by taking the cached
  // worst distance from any child nodes (Stat().FirstBound()).  This
  // corresponds roughly to B_1(N_q) in the paper.

  // The other way of bounding is to use the triangle inequality.  To do this,
  // we find the current best kth-neighbor candidate distance of any descendant
  // query point, and use the triangle inequality to place a bound on the
  // distance that candidate would have to any other descendant query point.
  // This corresponds roughly to B_2(N_q) in the paper, and is the bounding
  // style for cover trees.

  // Then, to assemble the final bound, since both bounds are valid, we simply
  // take the better of the two.

  double worstDistance = SortPolicy::BestDistance();
  double bestDistance = SortPolicy::WorstDistance();
  double bestPointDistance = SortPolicy::WorstDistance();
  double auxDistance = SortPolicy::WorstDistance();

  // Loop over points held in the node.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double distance = candidates[queryNode.Point(i)].top().first;
    if (SortPolicy::IsBetter(worstDistance, distance))
      worstDistance = distance;
    if (SortPolicy::IsBetter(distance, bestPointDistance))
      bestPointDistance = distance;
  }

  auxDistance = bestPointDistance;

  // Loop over children of the node, and use their cached information to
  // assemble bounds.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double firstBound = queryNode.Child(i).Stat().FirstBound();
    const double auxBound = queryNode.Child(i).Stat().AuxBound();

    if (SortPolicy::IsBetter(worstDistance, firstBound))
      worstDistance = firstBound;
    if (SortPolicy::IsBetter(auxBound, auxDistance))
      auxDistance = auxBound;
  }

  // Add triangle inequality adjustment to best distance.  It is possible this
  // could be tighter for some certain types of trees.
  bestDistance = SortPolicy::CombineWorst(auxDistance,
      2 * queryNode.FurthestDescendantDistance());

  // Add triangle inequality adjustment to best distance of points in node.
  bestPointDistance = SortPolicy::CombineWorst(bestPointDistance,
      queryNode.FurthestPointDistance() +
      queryNode.FurthestDescendantDistance());

  if (SortPolicy::IsBetter(bestPointDistance, bestDistance))
    bestDistance = bestPointDistance;

  // At this point:
  // worstDistance holds the value of B_1(N_q).
  // bestDistance holds the value of B_2(N_q).
  // auxDistance holds the value of B_aux(N_q).

  // Now consider the parent bounds.
  if (queryNode.Parent() != NULL)
  {
    // The parent's worst distance bound implies that the bound for this node
    // must be at least as good.  Thus, if the parent worst distance bound is
    // better, then take it.
    if (SortPolicy::IsBetter(queryNode.Parent()->Stat().FirstBound(),
        worstDistance))
      worstDistance = queryNode.Parent()->Stat().FirstBound();

    // The parent's best distance bound implies that the bound for this node
    // must be at least as good.  Thus, if the parent best distance bound is
    // better, then take it.
    if (SortPolicy::IsBetter(queryNode.Parent()->Stat().SecondBound(),
        bestDistance))
      bestDistance = queryNode.Parent()->Stat().SecondBound();
  }

  // Could the existing bounds be better?
  if (SortPolicy::IsBetter(queryNode.Stat().FirstBound(), worstDistance))
    worstDistance = queryNode.Stat().FirstBound();
  if (SortPolicy::IsBetter(queryNode.Stat().SecondBound(), bestDistance))
    bestDistance = queryNode.Stat().SecondBound();

  // Cache bounds for later.
  queryNode.Stat().FirstBound() = worstDistance;
  queryNode.Stat().SecondBound() = bestDistance;
  queryNode.Stat().AuxBound() = auxDistance;

  worstDistance = SortPolicy::Relax(worstDistance, epsilon);

  // We can't consider B_2 for Spill Trees.
  if (tree::IsSpillTree<TreeType>::value)
    return worstDistance;

  if (SortPolicy::IsBetter(worstDistance, bestDistance))
    return worstDistance;
  else
    return bestDistance;
}

/**
 * Helper function to insert a point into the list of candidate points.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
inline void NeighborSearchRules<SortPolicy, MetricType, TreeType>::
InsertNeighbor(
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

} // namespace neighbor
} // namespace mlpack

#endif // MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP
