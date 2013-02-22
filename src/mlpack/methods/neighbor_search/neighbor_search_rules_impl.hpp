/**
 * @file nearest_neighbor_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of NearestNeighborRules.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEAREST_NEIGHBOR_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "neighbor_search_rules.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearchRules<SortPolicy, MetricType, TreeType>::NeighborSearchRules(
    const arma::mat& referenceSet,
    const arma::mat& querySet,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances,
    MetricType& metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    neighbors(neighbors),
    distances(distances),
    metric(metric)
{ /* Nothing left to do. */ }

template<typename SortPolicy, typename MetricType, typename TreeType>
inline force_inline // Absolutely MUST be inline so optimizations can happen.
double NeighborSearchRules<SortPolicy, MetricType, TreeType>::
BaseCase(const size_t queryIndex, const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return 0.0;

  double distance = metric.Evaluate(querySet.unsafe_col(queryIndex),
                                    referenceSet.unsafe_col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distances.unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, distance);

  return distance;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Prescore(
    TreeType& queryNode,
    TreeType& referenceNode,
    TreeType& referenceChildNode,
    const double baseCaseResult) const
{
  const double distance = SortPolicy::BestNodeToNodeDistance(&queryNode,
      &referenceNode, &referenceChildNode, baseCaseResult);

  // Update our bound.
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestDistance = queryNode.Stat().Bound();

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::PrescoreQ(
    TreeType& queryNode,
    TreeType& queryChildNode,
    TreeType& referenceNode,
    const double baseCaseResult) const
{
  const double distance = SortPolicy::BestNodeToNodeDistance(&referenceNode,
      &queryNode, &queryChildNode, baseCaseResult);

  // Update our bound.
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestDistance = queryNode.Stat().Bound();

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode) const
{
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double distance = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode);
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double baseCaseResult) const
{
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double distance = SortPolicy::BestPointToNodeDistance(queryPoint,
      &referenceNode, baseCaseResult);
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
    TreeType& referenceNode) const
{
  const double distance = SortPolicy::BestNodeToNodeDistance(&queryNode,
      &referenceNode);

  // Update our bound.
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestDistance = queryNode.Stat().Bound();

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double baseCaseResult) const
{
  const double distance = SortPolicy::BestNodeToNodeDistance(&queryNode,
      &referenceNode, baseCaseResult);

  // Update our bound.
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestDistance = queryNode.Stat().Bound();

  return (SortPolicy::IsBetter(distance, bestDistance)) ? distance : DBL_MAX;
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
  queryNode.Stat().Bound() = CalculateBound(queryNode);
  const double bestDistance = queryNode.Stat().Bound();

  return (SortPolicy::IsBetter(oldScore, bestDistance)) ? oldScore : DBL_MAX;
}
/*
template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::FinishNode(
    TreeType& queryNode) const
{
  // Find the bound of points contained in this node.
  double pointBound = SortPolicy::BestDistance();
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    // The bound for this point is the k-th best distance plus the maximum
    // distance to a child of this node.
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i)) +
        maxDescendantDistance;
    if (SortPolicy::IsBetter(pointBound, bound))
      pointBound = bound;
  }

  // Push bound to parent.
}
*/

// Calculate the bound for a given query node in its current state.
template<typename SortPolicy, typename MetricType, typename TreeType>
inline double NeighborSearchRules<SortPolicy, MetricType, TreeType>::
    CalculateBound(TreeType& queryNode) const
{
  // We have four possible bounds, and we must take the best of them all.  We
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
  // (4) B(parent of queryNode);
  //
  // D_p[k] is the current k'th candidate distance for point p.
  // So we will loop over the points in queryNode and the children in queryNode
  // to calculate all four of these quantities.

  double worstPointDistance = SortPolicy::BestDistance();
  double bestPointDistance = SortPolicy::WorstDistance();

  // Loop over all points in this node to find the best and worst distance
  // candidates (for (1) and (2)).
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    const double distance = distances(distances.n_rows - 1, queryNode.Point(i));
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
    const double bound = queryNode.Child(i).Stat().Bound();
    const double childMaxDescendantDistance =
        queryNode.Child(i).FurthestDescendantDistance();

    if (SortPolicy::IsBetter(worstChildBound, bound))
      worstChildBound = bound;

    // Now calculate adjustment for maximum descendant distances.
    const double adjustedBound = SortPolicy::CombineWorst(bound,
        2 * (queryMaxDescendantDistance - childMaxDescendantDistance));
    if (SortPolicy::IsBetter(adjustedBound, bestAdjustedChildBound))
      bestAdjustedChildBound = adjustedBound;
  }

  // This is bound (1).
  const double firstBound =
      (SortPolicy::IsBetter(worstPointDistance, worstChildBound)) ?
      worstChildBound : worstPointDistance;

  // This is bound (2).
  const double secondBound = SortPolicy::CombineWorst(bestPointDistance,
      2 * queryMaxDescendantDistance);

  // Bound (3) is bestAdjustedChildBound.
  const double fourthBound = (queryNode.Parent() != NULL) ?
      queryNode.Parent()->Stat().Bound() : SortPolicy::WorstDistance();

  // Now, we will take the best of these.  Unfortunately due to the way
  // IsBetter() is defined, this sort of has to be a little ugly...
  const double interA = (SortPolicy::IsBetter(firstBound, secondBound)) ?
      firstBound : secondBound;
  const double interB =
      (SortPolicy::IsBetter(bestAdjustedChildBound, fourthBound)) ?
      bestAdjustedChildBound : fourthBound;

  return (SortPolicy::IsBetter(interA, interB)) ? interA : interB;
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
