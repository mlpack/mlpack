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
inline void NeighborSearchRules<SortPolicy, MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return;

  double distance = metric.Evaluate(querySet.col(queryIndex),
                                    referenceSet.col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distances.unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, distance);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline bool NeighborSearchRules<SortPolicy, MetricType, TreeType>::CanPrune(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  // Find the best distance between the query point and the node.
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double distance =
      SortPolicy::BestPointToNodeDistance(queryPoint, &referenceNode);
  const double bestDistance = distances(distances.n_rows - 1, queryIndex);

  // If this is better than the best distance we've seen so far, maybe there
  // will be something down this node.
  if (SortPolicy::IsBetter(distance, bestDistance))
    return false; // We cannot prune.
  else
    return true; // There cannot be anything better in this node.  So prune it.
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline bool NeighborSearchRules<SortPolicy, MetricType, TreeType>::CanPrune(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  const double distance = SortPolicy::BestNodeToNodeDistance(
      &queryNode, &referenceNode);
  const double bestDistance = queryNode.Stat().Bound();

  if (SortPolicy::IsBetter(distance, bestDistance))
    return false; // Can't prune.
  else
    return true;
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline bool NeighborSearchRules<SortPolicy, MetricType, TreeType>::LeftFirst(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  // This ends up with us calculating this distance twice (it will be done again
  // in CanPrune()), but because single-neighbors recursion is not the most
  // important in this method, we can let it slide.
  const arma::vec queryPoint = querySet.unsafe_col(queryIndex);
  const double leftDistance = SortPolicy::BestPointToNodeDistance(queryPoint,
      referenceNode.Left());
  const double rightDistance = SortPolicy::BestPointToNodeDistance(queryPoint,
      referenceNode.Right());

  return SortPolicy::IsBetter(leftDistance, rightDistance);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
inline bool NeighborSearchRules<SortPolicy, MetricType, TreeType>::LeftFirst(
    TreeType& staticNode,
    TreeType& recurseNode)
{
  const double leftDistance = SortPolicy::BestNodeToNodeDistance(&staticNode,
      recurseNode.Left());
  const double rightDistance = SortPolicy::BestNodeToNodeDistance(&staticNode,
      recurseNode.Right());

  return SortPolicy::IsBetter(leftDistance, rightDistance);
}

template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearchRules<
    SortPolicy,
    MetricType,
    TreeType>::
UpdateAfterRecursion(TreeType& queryNode, TreeType& /* referenceNode */)
{
  // Find the worst distance that the children found (including any points), and
  // update the bound accordingly.
  double worstDistance = SortPolicy::BestDistance();

  // First look through children nodes.
  for (size_t i = 0; i < queryNode.NumChildren(); ++i)
  {
    if (SortPolicy::IsBetter(worstDistance, queryNode.Child(i).Stat().Bound()))
      worstDistance = queryNode.Child(i).Stat().Bound();
  }

  // Now look through children points.
  for (size_t i = 0; i < queryNode.NumPoints(); ++i)
  {
    if (SortPolicy::IsBetter(worstDistance,
        distances(distances.n_rows - 1, queryNode.Point(i))))
      worstDistance = distances(distances.n_rows - 1, queryNode.Point(i));
  }

  // Take the worst distance from all of these, and update our bound to reflect
  // that.
  queryNode.Stat().Bound() = worstDistance;
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
