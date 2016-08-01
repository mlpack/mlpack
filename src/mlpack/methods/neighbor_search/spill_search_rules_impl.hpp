/**
 * @file spill_search_rules_impl.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Implementation of NeighborSearchRules for Spill Trees.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_SPILL_SEARCH_RULES_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_SPILL_SEARCH_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "spill_search_rules.hpp"

namespace mlpack {
namespace neighbor {

template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<MetricType,
    StatisticType, MatType, SplitType>>::NeighborSearchRules(
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

template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
void NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<MetricType,
    StatisticType, MatType, SplitType>>::GetResults(
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

template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
inline force_inline // Absolutely MUST be inline so optimizations can happen.
double NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<MetricType,
    StatisticType, MatType, SplitType>>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if (sameSet && (queryIndex == referenceIndex))
    return 0.0;

  double distance = metric.Evaluate(querySet.col(queryIndex),
                                    referenceSet.col(referenceIndex));
  ++baseCases;

  InsertNeighbor(queryIndex, referenceIndex, distance);

  return distance;
}

template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
inline double NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<
    MetricType, StatisticType, MatType, SplitType>>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  ++scores; // Count number of Score() calls.

  if (!referenceNode.Parent())
    return 0;

  if (referenceNode.Parent()->Overlap()) // Defeatist search.
  {
    const double value = referenceNode.Parent()->SplitValue();
    const size_t dim = referenceNode.Parent()->SplitDimension();
    const bool left = &referenceNode == referenceNode.Parent()->Left();

    if ((left && querySet(dim, queryIndex) <= value) ||
        (!left && querySet(dim, queryIndex) > value))
      return 0;
    else
      return DBL_MAX;
  }

  double distance = SortPolicy::BestPointToNodeDistance(
      querySet.col(queryIndex), &referenceNode);

  // Compare against the best k'th distance for this query point so far.
  double bestDistance = candidates[queryIndex].top().first;
  bestDistance = SortPolicy::Relax(bestDistance, epsilon);

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;
}

template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
inline double NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<
    MetricType, StatisticType, MatType, SplitType>>::Rescore(
    const size_t queryIndex,
    TreeType& /* referenceNode */,
    double oldScore) const
{
  // If we are already pruning, still prune.
  if (oldScore == DBL_MAX)
    return oldScore;

  // Just check the score again against the distances.
  double bestDistance = candidates[queryIndex].top().first;
  bestDistance = SortPolicy::Relax(bestDistance, epsilon);

  return (SortPolicy::IsBetter(oldScore, bestDistance)) ? oldScore : DBL_MAX;
}

template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
inline double NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<
    MetricType, StatisticType, MatType, SplitType>>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  ++scores; // Count number of Score() calls

  if (!referenceNode.Parent())
    return 0;

  if (referenceNode.Parent()->Overlap()) // Defeatist search.
  {
    const double value = referenceNode.Parent()->SplitValue();
    const size_t dim = referenceNode.Parent()->SplitDimension();
    const bool left = &referenceNode == referenceNode.Parent()->Left();

    if ((left && queryNode.Bound()[dim].Lo() <= value) ||
        (!left && queryNode.Bound()[dim].Hi() > value))
      return 0;
    else
      return DBL_MAX;
  }

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

  if (score == 0.0) // Nothing we can do here.
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
    // There isn't any need to set the traversal information because no
    // descendant combinations will be visited, and those are the only
    // combinations that would depend on the traversal information.
    return DBL_MAX;
  }

  double distance = SortPolicy::BestNodeToNodeDistance(&queryNode,
      &referenceNode);

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

template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
inline double NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<
    MetricType, StatisticType, MatType, SplitType>>::Rescore(
    TreeType& queryNode,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  if (oldScore == DBL_MAX)
    return oldScore;

  if (oldScore == SortPolicy::BestDistance())
    return oldScore;

  // Update our bound.
  const double bestDistance = CalculateBound(queryNode);

  return (SortPolicy::IsBetter(oldScore, bestDistance)) ? oldScore : DBL_MAX;
}

// Calculate the bound for a given query node in its current state and update
// it.
template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
inline double NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<
    MetricType, StatisticType, MatType, SplitType>>::
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

  double worstDistance = SortPolicy::BestDistance();

  // Loop over points held in the node.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double distance = candidates[queryNode.Point(i)].top().first;
    if (SortPolicy::IsBetter(worstDistance, distance))
      worstDistance = distance;
  }

  // Loop over children of the node, and use their cached information to
  // assemble bounds.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double firstBound = queryNode.Child(i).Stat().FirstBound();

    if (SortPolicy::IsBetter(worstDistance, firstBound))
      worstDistance = firstBound;
  }

  // At this point, worstDistance holds the value of B_1(N_q).

  // Now consider the parent bounds.
  if (queryNode.Parent() != NULL)
  {
    // The parent's worst distance bound implies that the bound for this node
    // must be at least as good.  Thus, if the parent worst distance bound is
    // better, then take it.
    if (SortPolicy::IsBetter(queryNode.Parent()->Stat().FirstBound(),
        worstDistance))
      worstDistance = queryNode.Parent()->Stat().FirstBound();
  }

  // Could the existing bounds be better?
  if (SortPolicy::IsBetter(queryNode.Stat().FirstBound(), worstDistance))
    worstDistance = queryNode.Stat().FirstBound();

  // Cache bounds for later.
  queryNode.Stat().FirstBound() = worstDistance;

  worstDistance = SortPolicy::Relax(worstDistance, epsilon);

  return worstDistance;
}

/**
 * Helper function to insert a point into the list of candidate points.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
inline void NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<
    MetricType, StatisticType, MatType, SplitType>>::InsertNeighbor(
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

#endif // MLPACK_METHODS_NEIGHBOR_SEARCH_SPILL_SEARCH_RULES_IMPL_HPP
