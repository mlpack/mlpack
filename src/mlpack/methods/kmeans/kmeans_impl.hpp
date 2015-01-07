/**
 * @file kmeans_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * Implementation for the K-means method for getting an initial point.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "kmeans.hpp"

#include <mlpack/core/tree/mrkd_statistic.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <stack>
#include <limits>

namespace mlpack {
namespace kmeans {

/**
 * Construct the K-Means object.
 */
template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy>
KMeans<
    MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::
KMeans(const size_t maxIterations,
       const double overclusteringFactor,
       const MetricType metric,
       const InitialPartitionPolicy partitioner,
       const EmptyClusterPolicy emptyClusterAction) :
    maxIterations(maxIterations),
    metric(metric),
    partitioner(partitioner),
    emptyClusterAction(emptyClusterAction)
{
  // Validate overclustering factor.
  if (overclusteringFactor < 1.0)
  {
    Log::Warn << "KMeans::KMeans(): overclustering factor must be >= 1.0 ("
        << overclusteringFactor << " given). Setting factor to 1.0.\n";
    this->overclusteringFactor = 1.0;
  }
  else
  {
    this->overclusteringFactor = overclusteringFactor;
  }
}

/**
 * Perform k-means clustering on the data, returning a list of cluster
 * assignments.  This just forward to the other function, which returns the
 * centroids too.  If this is properly inlined, there shouldn't be any
 * performance penalty whatsoever.
 */
template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy>
template<typename MatType>
inline void KMeans<
    MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::Col<size_t>& assignments,
        const bool initialGuess) const
{
  MatType centroids(data.n_rows, clusters);
  Cluster(data, clusters, assignments, centroids, initialGuess);
}

/**
 * Perform k-means clustering on the data, returning a list of cluster
 * assignments and the centroids of each cluster.
 */
template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy>
template<typename MatType>
void KMeans<
    MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::Col<size_t>& assignments,
        MatType& centroids,
        const bool initialAssignmentGuess,
        const bool initialCentroidGuess) const
{
  // Make sure we have more points than clusters.
  if (clusters > data.n_cols)
    Log::Warn << "KMeans::Cluster(): more clusters requested than points given."
        << std::endl;

  // Make sure our overclustering factor is valid.
  size_t actualClusters = size_t(overclusteringFactor * clusters);
  if (actualClusters > data.n_cols && overclusteringFactor != 1.0)
  {
    Log::Warn << "KMeans::Cluster(): overclustering factor is too large.  No "
        << "overclustering will be done." << std::endl;
    actualClusters = clusters;
  }

  // Now, the initial assignments.  First determine if they are necessary.
  if (initialAssignmentGuess)
  {
    if (assignments.n_elem != data.n_cols)
      Log::Fatal << "KMeans::Cluster(): initial cluster assignments (length "
          << assignments.n_elem << ") not the same size as the dataset (size "
          << data.n_cols << ")!" << std::endl;
  }
  else if (initialCentroidGuess)
  {
    if (centroids.n_cols != clusters)
      Log::Fatal << "KMeans::Cluster(): wrong number of initial cluster "
        << "centroids (" << centroids.n_cols << ", should be " << clusters
        << ")!" << std::endl;

    if (centroids.n_rows != data.n_rows)
      Log::Fatal << "KMeans::Cluster(): initial cluster centroids have wrong "
        << " dimensionality (" << centroids.n_rows << ", should be "
        << data.n_rows << ")!" << std::endl;

    // If there were no problems, construct the initial assignments from the
    // given centroids.
    assignments.set_size(data.n_cols);
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      // Find the closest centroid to this point.
      double minDistance = std::numeric_limits<double>::infinity();
      size_t closestCluster = clusters; // Invalid value.

      for (size_t j = 0; j < clusters; j++)
      {
        double distance = metric.Evaluate(data.col(i), centroids.col(j));

        if (distance < minDistance)
        {
          minDistance = distance;
          closestCluster = j;
        }
      }

      // Assign the point to the closest cluster that we found.
      assignments[i] = closestCluster;
    }
  }
  else
  {
    // Use the partitioner to come up with the partition assignments.
    partitioner.Cluster(data, actualClusters, assignments);
  }

  // Counts of points in each cluster.
  arma::Col<size_t> counts(actualClusters);
  counts.zeros();

  // Resize to correct size.
  centroids.set_size(data.n_rows, actualClusters);

  // Set counts correctly.
  for (size_t i = 0; i < assignments.n_elem; i++)
    counts[assignments[i]]++;

  size_t changedAssignments = 0;
  size_t iteration = 0;
  do
  {
    // Update step.
    // Calculate centroids based on given assignments.
    centroids.zeros();

    for (size_t i = 0; i < data.n_cols; i++)
      centroids.col(assignments[i]) += data.col(i);

    for (size_t i = 0; i < actualClusters; i++)
      centroids.col(i) /= counts[i];

    // Assignment step.
    // Find the closest centroid to each point.  We will keep track of how many
    // assignments change.  When no assignments change, we are done.
    changedAssignments = 0;
    for (size_t i = 0; i < data.n_cols; i++)
    {
      // Find the closest centroid to this point.
      double minDistance = std::numeric_limits<double>::infinity();
      size_t closestCluster = actualClusters; // Invalid value.

      for (size_t j = 0; j < actualClusters; j++)
      {
        double distance = metric.Evaluate(data.col(i), centroids.col(j));

        if (distance < minDistance)
        {
          minDistance = distance;
          closestCluster = j;
        }
      }

      // Reassign this point to the closest cluster.
      if (assignments[i] != closestCluster)
      {
        // Update counts.
        counts[assignments[i]]--;
        counts[closestCluster]++;
        // Update assignment.
        assignments[i] = closestCluster;
        changedAssignments++;
      }
    }

    // If we are not allowing empty clusters, then check that all of our
    // clusters have points.
    for (size_t i = 0; i < actualClusters; i++)
      if (counts[i] == 0)
        changedAssignments += emptyClusterAction.EmptyCluster(data, i,
            centroids, counts, assignments);

    iteration++;

  } while (changedAssignments > 0 && iteration != maxIterations);

  if (iteration != maxIterations)
  {
    Log::Debug << "KMeans::Cluster(): converged after " << iteration
        << " iterations." << std::endl;
  }
  else
  {
    Log::Debug << "KMeans::Cluster(): terminated after limit of " << iteration
        << " iterations." << std::endl;

    // Recalculate final clusters.
    centroids.zeros();

    for (size_t i = 0; i < data.n_cols; i++)
      centroids.col(assignments[i]) += data.col(i);

    for (size_t i = 0; i < actualClusters; i++)
      centroids.col(i) /= counts[i];
  }

  // If we have overclustered, we need to merge the nearest clusters.
  if (actualClusters != clusters)
  {
    // Generate a list of all the clusters' distances from each other.  This
    // list will become mangled and unused as the number of clusters decreases.
    size_t numDistances = ((actualClusters - 1) * actualClusters) / 2;
    size_t clustersLeft = actualClusters;
    arma::vec distances(numDistances);
    arma::Col<size_t> firstCluster(numDistances);
    arma::Col<size_t> secondCluster(numDistances);

    // Keep the mappings of clusters that we are changing.
    arma::Col<size_t> mappings = arma::linspace<arma::Col<size_t> >(0,
        actualClusters - 1, actualClusters);

    size_t i = 0;
    for (size_t first = 0; first < actualClusters; first++)
    {
      for (size_t second = first + 1; second < actualClusters; second++)
      {
        distances(i) = metric.Evaluate(centroids.col(first),
                                       centroids.col(second));
        firstCluster(i) = first;
        secondCluster(i) = second;
        i++;
      }
    }

    while (clustersLeft != clusters)
    {
      arma::uword minIndex;
      distances.min(minIndex);

      // Now we merge the clusters which that distance belongs to.
      size_t first = firstCluster(minIndex);
      size_t second = secondCluster(minIndex);
      for (size_t j = 0; j < assignments.n_elem; j++)
        if (assignments(j) == second)
          assignments(j) = first;

      // Now merge the centroids.
      centroids.col(first) *= counts[first];
      centroids.col(first) += (counts[second] * centroids.col(second));
      centroids.col(first) /= (counts[first] + counts[second]);

      // Update the counts.
      counts[first] += counts[second];
      counts[second] = 0;

      // Now update all the relevant distances.
      // First the distances where either cluster is the second cluster.
      for (size_t cluster = 0; cluster < second; cluster++)
      {
        // The offset is sum^n i - sum^(n - m) i, where n is actualClusters and
        // m is the cluster we are trying to offset to.
        size_t offset = (size_t) (((actualClusters - 1) * cluster)
            + (cluster - pow(cluster, 2.0)) / 2) - 1;

        // See if we need to update the distance from this cluster to the first
        // cluster.
        if (cluster < first)
        {
          // Make sure it isn't already DBL_MAX.
          if (distances(offset + (first - cluster)) != DBL_MAX)
            distances(offset + (first - cluster)) = metric.Evaluate(
                centroids.col(first), centroids.col(cluster));
        }

        distances(offset + (second - cluster)) = DBL_MAX;
      }

      // Now the distances where the first cluster is the first cluster.
      size_t offset = (size_t) (((actualClusters - 1) * first)
          + (first - pow(first, 2.0)) / 2) - 1;
      for (size_t cluster = first + 1; cluster < actualClusters; cluster++)
      {
        // Make sure it isn't already DBL_MAX.
        if (distances(offset + (cluster - first)) != DBL_MAX)
        {
          distances(offset + (cluster - first)) = metric.Evaluate(
              centroids.col(first), centroids.col(cluster));
        }
      }

      // Max the distance between the first and second clusters.
      distances(offset + (second - first)) = DBL_MAX;

      // Now max the distances for the second cluster (which no longer has
      // anything in it).
      offset = (size_t) (((actualClusters - 1) * second)
          + (second - pow(second, 2.0)) / 2) - 1;
      for (size_t cluster = second + 1; cluster < actualClusters; cluster++)
        distances(offset + (cluster - second)) = DBL_MAX;

      clustersLeft--;

      // Update the cluster mappings.
      mappings(second) = first;
      // Also update any mappings that were pointed at the previous cluster.
      for (size_t cluster = 0; cluster < actualClusters; cluster++)
        if (mappings(cluster) == second)
          mappings(cluster) = first;
    }

    // Now remap the mappings down to the smallest possible numbers.
    // Could this process be sped up?
    arma::Col<size_t> remappings(actualClusters);
    remappings.fill(actualClusters);
    size_t remap = 0; // Counter variable.
    for (size_t cluster = 0; cluster < actualClusters; cluster++)
    {
      // If the mapping of the current cluster has not been assigned a value
      // yet, we will assign it a cluster number.
      if (remappings(mappings(cluster)) == actualClusters)
      {
        remappings(mappings(cluster)) = remap;
        remap++;
      }
    }

    // Fix the assignments using the mappings we created.
    for (size_t j = 0; j < assignments.n_elem; j++)
      assignments(j) = remappings(mappings(assignments(j)));
  }
}

template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy>
std::string KMeans<MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy>::ToString() const
{
  std::ostringstream convert;
  convert << "KMeans [" << this << "]" << std::endl;
  convert << "  Overclustering Factor: " << overclusteringFactor <<std::endl;
  convert << "  Max Iterations: " << maxIterations <<std::endl;
  convert << "  Metric: " << std::endl;
  convert << mlpack::util::Indent(metric.ToString(),2);
  convert << std::endl;
  return convert.str();
}

}; // namespace kmeans
}; // namespace mlpack
