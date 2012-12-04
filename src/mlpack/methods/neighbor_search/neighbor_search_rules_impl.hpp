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

  // Calculate the bound on the fly.  This bound will be the minimum of
  // pointBound (the bounds given by the points in this node) and childBound
  // (the bounds given by the children of this node).
  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  // Find the bound of the points contained in this node.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    // The bound for this point is the k-th best distance plus the maximum
    // distance to a child of this node.
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i)) +
        maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  // Find the bound of the children.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // Update our bound.
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
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

  // Calculate the bound on the fly.  This bound will be the minimum of
  // pointBound (the bounds given by the points in this node) and childBound
  // (the bounds given by the children of this node).
  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  // Find the bound of the points contained in this node.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    // The bound for this point is the k-th best distance plus the maximum
    // distance to a child of this node.
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i)) +
        maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  // Find the bound of the children.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // Update our bound.
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
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

  // Calculate the bound on the fly.  This bound will be the minimum of
  // pointBound (the bounds given by the points in this node) and childBound
  // (the bounds given by the children of this node).
  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  // Find the bound of the points contained in this node.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    // The bound for this point is the k-th best distance plus the maximum
    // distance to a child of this node.
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i)) +
        maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  // Find the bound of the children.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // Update our bound.
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
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

  // Calculate the bound on the fly.  This bound will be the minimum of
  // pointBound (the bounds given by the points in this node) and childBound
  // (the bounds given by the children of this node).
  double pointBound = DBL_MAX;
  double childBound = DBL_MAX;
  const double maxDescendantDistance = queryNode.FurthestDescendantDistance();

  // Find the bound of the points contained in this node.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    // The bound for this point is the k-th best distance plus the maximum
    // distance to a child of this node.
    const double bound = distances(distances.n_rows - 1, queryNode.Point(i)) +
        maxDescendantDistance;
    if (bound < pointBound)
      pointBound = bound;
  }

  // Find the bound of the children.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    const double bound = queryNode.Child(i).Stat().Bound();
    if (bound < childBound)
      childBound = bound;
  }

  // Update our bound.
  queryNode.Stat().Bound() = std::min(pointBound, childBound);
  const double bestDistance = queryNode.Stat().Bound();

  return (SortPolicy::IsBetter(oldScore, bestDistance)) ? oldScore : DBL_MAX;
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
