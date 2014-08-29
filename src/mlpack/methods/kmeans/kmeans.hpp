/**
 * @file kmeans.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * K-Means clustering.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_KMEANS_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_KMEANS_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/metrics/lmetric.hpp>
#include "random_partition.hpp"
#include "max_variance_new_cluster.hpp"

#include <mlpack/core/tree/binary_space_tree.hpp>

namespace mlpack {
namespace kmeans /** K-Means clustering. */ {

/**
 * This class implements K-Means clustering.  This implementation supports
 * overclustering, which means that more clusters than are requested will be
 * found; then, those clusters will be merged together to produce the desired
 * number of clusters.
 *
 * Two template parameters can (optionally) be supplied: the policy for how to
 * find the initial partition of the data, and the actions to be taken when an
 * empty cluster is encountered, as well as the distance metric to be used.
 *
 * A simple example of how to run K-Means clustering is shown below.
 *
 * @code
 * extern arma::mat data; // Dataset we want to run K-Means on.
 * arma::Col<size_t> assignments; // Cluster assignments.
 *
 * KMeans<> k; // Default options.
 * k.Cluster(data, 3, assignments); // 3 clusters.
 *
 * // Cluster using the Manhattan distance, 100 iterations maximum, and an
 * // overclustering factor of 4.0.
 * KMeans<metric::ManhattanDistance> k(100, 4.0);
 * k.Cluster(data, 6, assignments); // 6 clusters.
 * @endcode
 *
 * @tparam MetricType The distance metric to use for this KMeans; see
 *     metric::LMetric for an example.
 * @tparam InitialPartitionPolicy Initial partitioning policy; must implement a
 *     default constructor and 'void Cluster(const arma::mat&, const size_t,
 *     arma::Col<size_t>&)'.
 * @tparam EmptyClusterPolicy Policy for what to do on an empty cluster; must
 *     implement a default constructor and 'void EmptyCluster(const arma::mat&,
 *     arma::Col<size_t&)'.
 *
 * @see RandomPartition, RefinedStart, AllowEmptyClusters, MaxVarianceNewCluster
 */
template<typename MetricType = metric::SquaredEuclideanDistance,
         typename InitialPartitionPolicy = RandomPartition,
         typename EmptyClusterPolicy = MaxVarianceNewCluster>
class KMeans
{
 public:
  /**
   * Create a K-Means object and (optionally) set the parameters which K-Means
   * will be run with.  This implementation allows a few strategies to improve
   * the performance of K-Means, including "overclustering" and disallowing
   * empty clusters.
   *
   * The overclustering factor controls how many clusters are
   * actually found; for instance, with an overclustering factor of 4, if
   * K-Means is run to find 3 clusters, it will actually find 12, then merge the
   * nearest clusters until only 3 are left.
   *
   * @param maxIterations Maximum number of iterations allowed before giving up
   *     (0 is valid, but the algorithm may never terminate).
   * @param overclusteringFactor Factor controlling how many extra clusters are
   *     found and then merged to get the desired number of clusters.
   * @param metric Optional MetricType object; for when the metric has state
   *     it needs to store.
   * @param partitioner Optional InitialPartitionPolicy object; for when a
   *     specially initialized partitioning policy is required.
   * @param emptyClusterAction Optional EmptyClusterPolicy object; for when a
   *     specially initialized empty cluster policy is required.
   */
  KMeans(const size_t maxIterations = 1000,
         const double overclusteringFactor = 1.0,
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
  template<typename MatType>
  void Cluster(const MatType& data,
               const size_t clusters,
               arma::Col<size_t>& assignments,
               const bool initialGuess = false) const;

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
   * Note that if the overclustering factor is greater than 1, the centroids
   * matrix will be resized in the method.  Regardless of the overclustering
   * factor, the centroid guess matrix (if initialCentroidGuess is set to true)
   * should have the same number of rows as the data matrix, and number of
   * columns equal to 'clusters'.
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
  template<typename MatType>
  void Cluster(const MatType& data,
               const size_t clusters,
               arma::Col<size_t>& assignments,
               MatType& centroids,
               const bool initialAssignmentGuess = false,
               const bool initialCentroidGuess = false) const;

  //! Return the overclustering factor.
  double OverclusteringFactor() const { return overclusteringFactor; }
  //! Set the overclustering factor.  Must be greater than 1.
  double& OverclusteringFactor() { return overclusteringFactor; }

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

  // Returns a string representation of this object.
  std::string ToString() const;

 private:
  //! Factor controlling how many clusters are actually found.
  double overclusteringFactor;
  //! Maximum number of iterations before giving up.
  size_t maxIterations;
  //! Instantiated distance metric.
  MetricType metric;
  //! Instantiated initial partitioning policy.
  InitialPartitionPolicy partitioner;
  //! Instantiated empty cluster policy.
  EmptyClusterPolicy emptyClusterAction;
};

}; // namespace kmeans
}; // namespace mlpack

// Include implementation.
#include "kmeans_impl.hpp"

#endif // __MLPACK_METHODS_MOG_KMEANS_HPP
