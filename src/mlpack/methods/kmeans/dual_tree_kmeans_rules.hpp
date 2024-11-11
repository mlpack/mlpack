/**
 * @file methods/kmeans/dual_tree_kmeans_rules.hpp
 * @author Ryan Curtin
 *
 * A set of rules for the dual-tree k-means algorithm which uses dual-tree
 * nearest neighbor search.  For the most part we'll call out to
 * NeighborSearchRules when we can.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_HPP
#define MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_HPP

#include <mlpack/core/tree/traversal_info.hpp>

namespace mlpack {

template<typename DistanceType, typename TreeType>
class DualTreeKMeansRules
{
 public:
  DualTreeKMeansRules(const arma::mat& centroids,
                      const arma::mat& dataset,
                      arma::Row<size_t>& assignments,
                      arma::vec& upperBounds,
                      arma::vec& lowerBounds,
                      DistanceType& distance,
                      const std::vector<bool>& prunedPoints,
                      const std::vector<size_t>& oldFromNewCentroids,
                      std::vector<bool>& visited);

  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  double Score(const size_t queryIndex, TreeType& referenceNode);
  double Score(TreeType& queryNode, TreeType& referenceNode);
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore);
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore);

  using TraversalInfoType = mlpack::TraversalInfo<TreeType>;

  TraversalInfoType& TraversalInfo() { return traversalInfo; }
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }

  size_t BaseCases() const { return baseCases; }
  size_t& BaseCases() { return baseCases; }

  size_t Scores() const { return scores; }
  size_t& Scores() { return scores; }

  //! Get the minimum number of base cases needed for correct results for each
  //! query point.  This only matters in defeatist search mode.
  size_t MinimumBaseCases() const { return 0; }

 private:
  const arma::mat& centroids;
  const arma::mat& dataset;
  arma::Row<size_t>& assignments;
  arma::vec& upperBounds;
  arma::vec& lowerBounds;
  DistanceType& distance;

  const std::vector<bool>& prunedPoints;

  const std::vector<size_t>& oldFromNewCentroids;

  std::vector<bool>& visited;

  size_t baseCases;
  size_t scores;

  TraversalInfoType traversalInfo;

  size_t lastQueryIndex;
  size_t lastReferenceIndex;
  size_t lastBaseCase;
};

} // namespace mlpack

#include "dual_tree_kmeans_rules_impl.hpp"

#endif
