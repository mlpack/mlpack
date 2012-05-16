/**
 * @file neighbor_search_cover_impl.hpp
 * @author Ryan Curtin
 *
 * Cover tree implementation.  Ugly.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_COVER_RULES_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_COVER_RULES_IMPL_HPP

// In case it has not been included yet.
#include "neighbor_search_cover_rules_impl.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearchCoverRules<SortPolicy, MetricType, TreeType>::
    NeighborSearchCoverRules(const arma::mat& referenceSet,
                             const arma::mat& querySet,
                             arma::Mat<size_t>& neighbors,
                             arma::mat& distances,
                             MetricType& metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    neighbors(neighbors),
    distances(distances),
    metric(metric)
{ /* Nothing to do. */ }

template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearchCoverRules<SortPolicy, MetricType, TreeType>::BaseCase(
    const size_t queryIndex, const size_t referenceIndex)
{
  // Should be the same as the regular base case (doesn't depend on tree type).

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
bool NeighborSearchCoverRules<SortPolicy, MetricType, TreeType>::CanPrune(
    TreeType& queryNode, TreeType& referenceNode)
{
  // This is different than the regular implementation.  For query node at scale
  // j and reference node at scale i, the distance must be "better" than the
  // current worst best distance for the point of the query node minus 2^(j + 2)
  // and 2^(i + 1).

  // The distance we have to be better than.
  const double bestDistance = distances(distances.n_rows - 1,
      queryNode.Point());
  // Hard-coded nearest neighbor search.  Ignores SortPolicy... not good.
  double distance = metric.Evaluate(querySet.unsafe_col(queryNode.Point()),
      referenceSet.unsafe_col(referenceNode.Point()));

  if (referenceNode.Scale() != INT_MIN)
    distance -= std::pow(referenceNode.ExpansionConstant(),
        referenceNode.Scale() + 1);

  if (queryNode.Scale() != INT_MIN)
    distance -= std::pow(queryNode.ExpansionConstant(), queryNode.Scale() + 2);

  distance = std::max(distance, 0.0);

  // Suddenly not ignoring SortPolicy... ha!
  if (SortPolicy::IsBetter(distance, bestDistance))
    return false; // Can't prune.
  else
    return true; // Can prune.  No possibility of betterment.
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
void NeighborSearchCoverRules<SortPolicy, MetricType, TreeType>::InsertNeighbor(
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

#endif
