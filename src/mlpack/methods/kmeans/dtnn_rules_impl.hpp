/**
 * @file dtnn_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of DualTreeKMeansRules.
 */
#ifndef __MLPACK_METHODS_KMEANS_DTNN_RULES_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_DTNN_RULES_IMPL_HPP

#include "dtnn_rules.hpp"

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename TreeType>
DTNNKMeansRules<MetricType, TreeType>::DTNNKMeansRules(
    const arma::mat& centroids,
    const arma::mat& dataset,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances,
    MetricType& metric,
    const std::vector<bool>& prunedPoints,
    const std::vector<size_t>& oldFromNewCentroids) :
    neighbor::NeighborSearchRules<neighbor::NearestNeighborSort, MetricType,
        TreeType>(centroids, dataset, neighbors, distances, metric),
    prunedPoints(prunedPoints),
    oldFromNewCentroids(oldFromNewCentroids)
{
  // Nothing to do.
}

template<typename MetricType, typename TreeType>
inline force_inline double DTNNKMeansRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // We'll check if the query point has been pruned.  If so, don't continue.
  if (prunedPoints[queryIndex])
    return 0.0; // Returning 0 shouldn't be a problem.

  // This is basically an inlined NeighborSearchRules::BaseCase(), but it
  // differs in that it applies the mappings to the results automatically.
  // We can also skip a check or two.

  // By the way, all of the this-> is necessary because the parent class is a
  // dependent name, so all of the members of that parent aren't resolvable
  // before type substitution.  The 'this->' turns that member into a dependent
  // name too (since the type of 'this' is dependent), and thus the compiler
  // resolves the name later and we get no error.  Hooray C++!
  //
  // See also:
  // http://stackoverflow.com/questions/10639053/name-lookups-in-c-templates

  // If we have already performed this base case, do not perform it again.
  if ((this->lastQueryIndex == queryIndex) &&
      (this->lastReferenceIndex == referenceIndex))
    return this->lastBaseCase;

  double distance = this->metric.Evaluate(this->querySet.col(queryIndex),
      this->referenceSet.col(referenceIndex));
  ++this->baseCases;

  const size_t cluster = oldFromNewCentroids[referenceIndex];

  // Is this better than either existing candidate?
  if (distance < this->distances(0, queryIndex))
  {
    // Do we need to replace the assignment, or is it an old assignment from a
    // previous iteration?
    if (this->neighbors(0, queryIndex) != cluster &&
        this->neighbors(0, queryIndex) < this->referenceSet.n_cols)
    {
      // We must push the old closest assignment down the stack.
      this->neighbors(1, queryIndex) = this->neighbors(0, queryIndex);
      this->distances(1, queryIndex) = this->distances(0, queryIndex);
      this->neighbors(0, queryIndex) = cluster;
    }
    else if (this->neighbors(0, queryIndex) >= this->referenceSet.n_cols)
    {
      this->neighbors(0, queryIndex) = cluster;
    }

    this->distances(0, queryIndex) = distance;
  }
  else if (distance < this->distances(1, queryIndex))
  {
    // Here it doesn't actually matter if the assignment is the same.
    this->neighbors(1, queryIndex) = cluster;
    this->distances(1, queryIndex) = distance;
  }

  this->lastQueryIndex = queryIndex;
  this->lastReferenceIndex = referenceIndex;
  this->lastBaseCase = distance;

  return distance;
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  // If the query point has already been pruned, then don't recurse further.
  if (prunedPoints[queryIndex])
    return DBL_MAX;

  return neighbor::NeighborSearchRules<neighbor::NearestNeighborSort,
      MetricType, TreeType>::Score(queryIndex, referenceNode);
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  if (queryNode.Stat().Pruned())
    return DBL_MAX;

  // Check if the query node is Hamerly pruned, and if not, then don't continue.
  return neighbor::NeighborSearchRules<neighbor::NearestNeighborSort,
      MetricType, TreeType>::Score(queryNode, referenceNode);
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Rescore(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double oldScore)
{
  return neighbor::NeighborSearchRules<neighbor::NearestNeighborSort,
      MetricType, TreeType>::Rescore(queryIndex, referenceNode, oldScore);
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Rescore(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double oldScore)
{
  // No need to check for a Hamerly prune.  Because we've already done that in
  // Score().
  return neighbor::NeighborSearchRules<neighbor::NearestNeighborSort,
      MetricType, TreeType>::Rescore(queryNode, referenceNode, oldScore);
}

} // namespace kmeans
} // namespace mlpack

#endif
