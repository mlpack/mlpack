/**
 * @file kmeans_rules.hpp
 * @author Ryan Curtin
 *
 * Defines the pruning rules and base cases rules necessary to perform
 * single-tree k-means clustering using the Pelleg-Moore fast k-means algorithm,
 * which has been shoehorned to fit into the mlpack tree abstractions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_HPP
#define MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_HPP

namespace mlpack {
namespace kmeans {

/**
 * The rules class for the single-tree Pelleg-Moore kd-tree traversal for
 * k-means clustering.  Although TreeType is a free template parameter, this
 * particular implementation is specialized to trees with hyper-rectangle bounds
 * due to the pruning rule used to determine if one cluster dominates a node
 * with respect to another cluster.
 *
 * Our implementation here abuses the single-tree algorithm abstractions a
 * little bit.  Instead of doing a traversal for a particular query point, in
 * this case we consider all clusters at once---so the query point is entirely
 * ignored during in BaseCase() and Score().
 */
template<typename MetricType, typename TreeType>
class PellegMooreKMeansRules
{
 public:
  /**
   * Create the PellegMooreKMeansRules object.
   *
   * @param dataset The dataset that the tree is built on.
   * @param centroids The current centroids.
   * @param newCentroids New centroids after this iteration (output).
   * @param counts Current cluster counts, to be replaced with new cluster
   *      counts.
   * @param metric Instantiated metric.
   */
  PellegMooreKMeansRules(const typename TreeType::Mat& dataset,
                         const arma::mat& centroids,
                         arma::mat& newCentroids,
                         arma::Col<size_t>& counts,
                         MetricType& metric);

  /**
   * The BaseCase() function for this single-tree algorithm does nothing.
   * Instead, point-to-cluster comparisons are handled as necessary in Score().
   *
   * @param queryIndex Index of query point (fake, will be ignored).
   * @param referenceIndex Index of reference point.
   */
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  /**
   * Determine if a cluster can be pruned, and if not, perform point-to-cluster
   * comparisons.  The point-to-cluster comparisons are performed here and not
   * in BaseCase() because of the complexity of managing the blacklist.
   *
   * @param queryIndex Index of query point (fake, will be ignored).
   * @param referenceNode Node containing points in the dataset.
   */
  double Score(const size_t queryIndex, TreeType& referenceNode);

  /**
   * Rescore to determine if a node can be pruned.  In this case, a node can
   * never be pruned during rescoring, so this just returns oldScore.
   *
   * @param queryIndex Index of query point (fake, will be ignored).
   * @param referenceNode Node containing points in the dataset.
   * @param oldScore Resulting score from Score().
   */
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore);

  //! Get the number of distance calculations that have been performed.
  size_t DistanceCalculations() const { return distanceCalculations; }
  //! Modify the number of distance calculations that have been performed.
  size_t& DistanceCalculations() { return distanceCalculations; }

 private:
  //! The dataset.
  const typename TreeType::Mat& dataset;
  //! The clusters.
  const arma::mat& centroids;
  //! The new centroids.
  arma::mat& newCentroids;
  //! The counts of points in each cluster.
  arma::Col<size_t>& counts;
  //! Instantiated metric.
  MetricType& metric;

  //! The number of O(d) distance calculations that have been performed.
  size_t distanceCalculations;
};

} // namespace kmeans
} // namespace mlpack

// Include implementation.
#include "pelleg_moore_kmeans_rules_impl.hpp"

#endif
