/**
 * @file kmeans_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * Implementation for the K-means method for getting an initial point.
 */
#include "kmeans.hpp"

#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kmeans {

/**
 * Construct the K-Means object.
 */
template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
KMeans<
    MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::
KMeans(const size_t maxIterations,
       const MetricType metric,
       const InitialPartitionPolicy partitioner,
       const EmptyClusterPolicy emptyClusterAction) :
    maxIterations(maxIterations),
    metric(metric),
    partitioner(partitioner),
    emptyClusterAction(emptyClusterAction)
{
  // Nothing to do.
}

/**
 * Perform k-means clustering on the data, returning a list of cluster
 * assignments.  This just forward to the other function, which returns the
 * centroids too.  If this is properly inlined, there shouldn't be any
 * performance penalty whatsoever.
 */
template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
inline void KMeans<
    MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::Col<size_t>& assignments,
        const bool initialGuess)
{
  arma::mat centroids(data.n_rows, clusters);
  Cluster(data, clusters, assignments, centroids, initialGuess);
}

/**
 * Perform k-means clustering on the data, returning a list of cluster
 * assignments and the centroids of each cluster.
 */
template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
void KMeans<
    MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::mat& centroids,
        const bool initialGuess)
{
  // Make sure we have more points than clusters.
  if (clusters > data.n_cols)
    Log::Warn << "KMeans::Cluster(): more clusters requested than points given."
        << std::endl;
  else if (clusters == 0)
    Log::Warn << "KMeans::Cluster(): zero clusters requested.  This probably "
        << "isn't going to work.  Brace for crash." << std::endl;

  // Check validity of initial guess.
  if (initialGuess)
  {
    if (centroids.n_cols != clusters)
      Log::Fatal << "KMeans::Cluster(): wrong number of initial cluster "
        << "centroids (" << centroids.n_cols << ", should be " << clusters
        << ")!" << std::endl;

    if (centroids.n_rows != data.n_rows)
      Log::Fatal << "KMeans::Cluster(): initial cluster centroids have wrong "
        << " dimensionality (" << centroids.n_rows << ", should be "
        << data.n_rows << ")!" << std::endl;
  }

  // Use the partitioner to come up with the partition assignments and calculate
  // the initial centroids.
  if (!initialGuess)
  {
    // The partitioner gives assignments, so we need to calculate centroids from
    // those assignments.  This is probably not the most efficient way to do
    // this, so maybe refactoring should be considered in the future.
    arma::Col<size_t> assignments;
    partitioner.Cluster(data, clusters, assignments);

    // Calculate initial centroids.
    arma::Col<size_t> counts;
    counts.zeros(clusters);
    centroids.zeros(data.n_rows, clusters);
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      centroids.col(assignments[i]) += arma::vec(data.col(i));
      counts[assignments[i]]++;
    }

    for (size_t i = 0; i < clusters; ++i)
      if (counts[i] != 0)
        centroids.col(i) /= counts[i];
  }

  // Counts of points in each cluster.
  arma::Col<size_t> counts(clusters);

  size_t iteration = 0;

  LloydStepType<MetricType, MatType> lloydStep(data, metric);
  arma::mat centroidsOther;
  double cNorm;

  do
  {
    // We have two centroid matrices.  We don't want to copy anything, so,
    // depending on the iteration number, we use a different centroid matrix...
    if (iteration % 2 == 0)
      cNorm = lloydStep.Iterate(centroids, centroidsOther, counts);
    else
      cNorm = lloydStep.Iterate(centroidsOther, centroids, counts);

    // If we are not allowing empty clusters, then check that all of our
    // clusters have points.
    for (size_t i = 0; i < clusters; i++)
    {
      if (counts[i] == 0)
      {
        Log::Info << "Cluster " << i << " is empty.\n";
        if (iteration % 2 == 0)
          emptyClusterAction.EmptyCluster(data, i, centroidsOther, counts,
              metric, iteration);
        else
          emptyClusterAction.EmptyCluster(data, i, centroids, counts, metric,
              iteration);
      }
    }

    iteration++;
    Log::Info << "KMeans::Cluster(): iteration " << iteration << ", residual "
        << cNorm << ".\n";

  } while (cNorm > 1e-5 && iteration != maxIterations);

  // If we ended on an even iteration, then the centroids are in the
  // centroidsOther matrix, and we need to steal its memory (steal_mem() avoids
  // a copy if possible).
  if ((iteration - 1) % 2 == 0)
    centroids.steal_mem(centroidsOther);

  if (iteration != maxIterations)
  {
    Log::Info << "KMeans::Cluster(): converged after " << iteration
        << " iterations." << std::endl;
  }
  else
  {
    Log::Info << "KMeans::Cluster(): terminated after limit of " << iteration
        << " iterations." << std::endl;
  }
  Log::Info << lloydStep.DistanceCalculations() << " distance calculations."
      << std::endl;
}

/**
 * Perform k-means clustering on the data, returning a list of cluster
 * assignments and the centroids of each cluster.
 */
template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
void KMeans<
    MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::Col<size_t>& assignments,
        arma::mat& centroids,
        const bool initialAssignmentGuess,
        const bool initialCentroidGuess)
{
  // Now, the initial assignments.  First determine if they are necessary.
  if (initialAssignmentGuess)
  {
    if (assignments.n_elem != data.n_cols)
      Log::Fatal << "KMeans::Cluster(): initial cluster assignments (length "
          << assignments.n_elem << ") not the same size as the dataset (size "
          << data.n_cols << ")!" << std::endl;

    // Calculate initial centroids.
    arma::Col<size_t> counts;
    counts.zeros(clusters);
    centroids.zeros(data.n_rows, clusters);
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      centroids.col(assignments[i]) += arma::vec(data.col(i));
      counts[assignments[i]]++;
    }

    for (size_t i = 0; i < clusters; ++i)
      if (counts[i] != 0)
        centroids.col(i) /= counts[i];
  }

  Cluster(data, clusters, centroids,
      initialAssignmentGuess || initialCentroidGuess);

  // Calculate final assignments.
  assignments.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Find the closest centroid to this point.
    double minDistance = std::numeric_limits<double>::infinity();
    size_t closestCluster = centroids.n_cols; // Invalid value.

    for (size_t j = 0; j < centroids.n_cols; j++)
    {
      const double distance = metric.Evaluate(data.col(i), centroids.col(j));

      if (distance < minDistance)
      {
        minDistance = distance;
        closestCluster = j;
      }
    }

    Log::Assert(closestCluster != centroids.n_cols);
    assignments[i] = closestCluster;
  }
}

template<typename MetricType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
std::string KMeans<MetricType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::ToString() const
{
  std::ostringstream convert;
  convert << "KMeans [" << this << "]" << std::endl;
  convert << "  Max Iterations: " << maxIterations << std::endl;
  convert << "  Metric: " << std::endl;
  convert << mlpack::util::Indent(metric.ToString(), 2);
  convert << std::endl;
  return convert.str();
}

}; // namespace kmeans
}; // namespace mlpack
