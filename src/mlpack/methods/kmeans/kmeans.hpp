/**
 * @file kmeans.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * K-Means clustering.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_KMEANS_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_KMEANS_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/metrics/lmetric.hpp>
#include "random_partition.hpp"
#include "max_variance_new_cluster.hpp"
#include "naive_kmeans.hpp"

#include <mlpack/core/tree/binary_space_tree.hpp>

namespace mlpack {
namespace kmeans /** K-Means clustering. */ {

/**
 * This class implements K-Means clustering, using a variety of possible
 * implementations of Lloyd's algorithm.
 *
 * Four template parameters can (optionally) be supplied: the distance metric to
 * use, the policy for how to find the initial partition of the data, the
 * actions to be taken when an empty cluster is encountered, and the
 * implementation of a single Lloyd step to use.
 *
 * A simple example of how to run K-Means clustering is shown below.
 *
 * @code
 * extern arma::mat data; // Dataset we want to run K-Means on.
 * arma::Row<size_t> assignments; // Cluster assignments.
 * arma::mat centroids; // Cluster centroids.
 *
 * KMeans<> k; // Default options.
 * k.Cluster(data, 3, assignments, centroids); // 3 clusters.
 *
 * // Cluster using the Manhattan distance, 100 iterations maximum, saving only
 * // the centroids.
 * KMeans<metric::ManhattanDistance> k(100);
 * k.Cluster(data, 6, centroids); // 6 clusters.
 * @endcode
 *
 * @tparam MetricType The distance metric to use for this KMeans; see
 *     metric::LMetric for an example.
 * @tparam InitialPartitionPolicy Initial partitioning policy; must implement a
 *     default constructor and 'void Cluster(const arma::mat&, const size_t,
 *     arma::Row<size_t>&)'.
 * @tparam EmptyClusterPolicy Policy for what to do on an empty cluster; must
 *     implement a default constructor and 'void EmptyCluster(const arma::mat&
 *     data, const size_t emptyCluster, const arma::mat& oldCentroids,
 *     arma::mat& newCentroids, arma::Col<size_t>& counts, MetricType& metric,
 *     const size_t iteration)'.
 * @tparam LloydStepType Implementation of single Lloyd step to use.
 *
 * @see RandomPartition, RefinedStart, AllowEmptyClusters,
 *      MaxVarianceNewCluster, NaiveKMeans, ElkanKMeans
 */
template<typename MetricType = metric::EuclideanDistance,
         typename InitialPartitionPolicy = RandomPartition,
         typename EmptyClusterPolicy = MaxVarianceNewCluster,
         template<class, class> class LloydStepType = NaiveKMeans,
         typename MatType = arma::mat>
class KMeans
{
 public:
  /**
   * Create a K-Means object and (optionally) set the parameters which K-Means
   * will be run with.
   *
   * @param maxIterations Maximum number of iterations allowed before giving up
   *     (0 is valid, but the algorithm may never terminate).
   * @param metric Optional MetricType object; for when the metric has state
   *     it needs to store.
   * @param partitioner Optional InitialPartitionPolicy object; for when a
   *     specially initialized partitioning policy is required.
   * @param emptyClusterAction Optional EmptyClusterPolicy object; for when a
   *     specially initialized empty cluster policy is required.
   */
  KMeans(const size_t maxIterations = 1000,
         const MetricType metric = MetricType(),
         const InitialPartitionPolicy partitioner = InitialPartitionPolicy(),
         const EmptyClusterPolicy emptyClusterAction = EmptyClusterPolicy());


  /**
   * Perform k-means clustering on the data, returning a list of cluster
   * assignments.  Optionally, the vector of assignments can be set to an
   * initial guess of the cluster assignments; to do this, set initialGuess to
   * true.
   *
   * @tparam MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param clusters Number of clusters to compute.
   * @param assignments Vector to store cluster assignments in.
   * @param initialGuess If true, then it is assumed that assignments has a list
   *      of initial cluster assignments.
   */
  void Cluster(const MatType& data,
               const size_t clusters,
               arma::Row<size_t>& assignments,
               const bool initialGuess = false);

  /**
   * Perform k-means clustering on the data, returning the centroids of each
   * cluster in the centroids matrix.  Optionally, the initial centroids can be
   * specified by filling the centroids matrix with the initial centroids and
   * specifying initialGuess = true.
   *
   * @tparam MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param clusters Number of clusters to compute.
   * @param centroids Matrix in which centroids are stored.
   * @param initialGuess If true, then it is assumed that centroids contains the
   *      initial cluster centroids.
   */
  void Cluster(const MatType& data,
               const size_t clusters,
               arma::mat& centroids,
               const bool initialGuess = false);

  /**
   * Perform k-means clustering on the data, returning a list of cluster
   * assignments and also the centroids of each cluster.  Optionally, the vector
   * of assignments can be set to an initial guess of the cluster assignments;
   * to do this, set initialAssignmentGuess to true.  Another way to set initial
   * cluster guesses is to fill the centroids matrix with the centroid guesses,
   * and then set initialCentroidGuess to true.  initialAssignmentGuess
   * supersedes initialCentroidGuess, so if both are set to true, the
   * assignments vector is used.
   *
   * @tparam MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param clusters Number of clusters to compute.
   * @param assignments Vector to store cluster assignments in.
   * @param centroids Matrix in which centroids are stored.
   * @param initialAssignmentGuess If true, then it is assumed that assignments
   *      has a list of initial cluster assignments.
   * @param initialCentroidGuess If true, then it is assumed that centroids
   *      contains the initial centroids of each cluster.
   */
  void Cluster(const MatType& data,
               const size_t clusters,
               arma::Row<size_t>& assignments,
               arma::mat& centroids,
               const bool initialAssignmentGuess = false,
               const bool initialCentroidGuess = false);

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Set the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the distance metric.
  const MetricType& Metric() const { return metric; }
  //! Modify the distance metric.
  MetricType& Metric() { return metric; }

  //! Get the initial partitioning policy.
  const InitialPartitionPolicy& Partitioner() const { return partitioner; }
  //! Modify the initial partitioning policy.
  InitialPartitionPolicy& Partitioner() { return partitioner; }

  //! Get the empty cluster policy.
  const EmptyClusterPolicy& EmptyClusterAction() const
  { return emptyClusterAction; }
  //! Modify the empty cluster policy.
  EmptyClusterPolicy& EmptyClusterAction() { return emptyClusterAction; }

  //! Serialize the k-means object.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);

 private:
  //! Maximum number of iterations before giving up.
  size_t maxIterations;
  //! Instantiated distance metric.
  MetricType metric;
  //! Instantiated initial partitioning policy.
  InitialPartitionPolicy partitioner;
  //! Instantiated empty cluster policy.
  EmptyClusterPolicy emptyClusterAction;
};

} // namespace kmeans
} // namespace mlpack

// Include implementation.
#include "kmeans_impl.hpp"

#endif // __MLPACK_METHODS_KMEANS_KMEANS_HPP
