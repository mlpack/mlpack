/**
 * @file methods/kmeans/kmeans_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * Implementation for the K-means method for getting an initial point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "kmeans.hpp"

#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>
#include <mlpack/core/util/size_checks.hpp>

namespace mlpack {

/**
 * This gives us a GivesCentroids object that we can use to tell whether or not
 * an InitialPartitionPolicy returns centroids or point assignments.
 */
HAS_MEM_FUNC(Cluster, GivesCentroidsCheck);

/**
 * 'value' is true if the InitialPartitionPolicy class has a member
 * Cluster(const arma::mat& data, const size_t clusters, arma::mat& centroids).
 */
template<typename InitialPartitionPolicy>
struct GivesCentroids
{
  static const bool value =
    // Non-static version.
    GivesCentroidsCheck<InitialPartitionPolicy,
        void(InitialPartitionPolicy::*)(const arma::mat&,
                                        const size_t,
                                        arma::mat&)>::value ||
    // Static version.
    GivesCentroidsCheck<InitialPartitionPolicy,
        void(*)(const arma::mat&, const size_t, arma::mat&)>::value;
};

//! Call the initial partition policy, if it returns assignments.  This returns
//! 'true' to indicate that assignments were given.
template<typename MatType,
         typename InitialPartitionPolicy>
bool GetInitialAssignmentsOrCentroids(
    InitialPartitionPolicy& ipp,
    const MatType& data,
    const size_t clusters,
    arma::Row<size_t>& assignments,
    arma::mat& /* centroids */,
    const typename std::enable_if_t<
        !GivesCentroids<InitialPartitionPolicy>::value>* = 0)
{
  ipp.Cluster(data, clusters, assignments);

  return true;
}

//! Call the initial partition policy, if it returns centroids.  This returns
//! 'false' to indicate that assignments were not given.
template<typename MatType,
         typename InitialPartitionPolicy>
bool GetInitialAssignmentsOrCentroids(
    InitialPartitionPolicy& ipp,
    const MatType& data,
    const size_t clusters,
    arma::Row<size_t>& /* assignments */,
    arma::mat& centroids,
    const typename std::enable_if_t<
        GivesCentroids<InitialPartitionPolicy>::value>* = 0)
{
  ipp.Cluster(data, clusters, centroids);

  return false;
}

/**
 * Construct the K-Means object.
 */
template<typename DistanceType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
KMeans<
    DistanceType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::
KMeans(const size_t maxIterations,
       const DistanceType distance,
       const InitialPartitionPolicy partitioner,
       const EmptyClusterPolicy emptyClusterAction) :
    maxIterations(maxIterations),
    distance(distance),
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
template<typename DistanceType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
inline void KMeans<
    DistanceType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::Row<size_t>& assignments,
        const bool initialGuess)
{
  arma::mat centroids(data.n_rows, clusters);
  Cluster(data, clusters, assignments, centroids, initialGuess);
}

/**
 * Perform k-means clustering on the data, returning a list of cluster
 * assignments and the centroids of each cluster.
 */
template<typename DistanceType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
void KMeans<
    DistanceType,
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
    util::CheckSameSizes(centroids, clusters, "KMeans::Cluster()", "clusters");
    util::CheckSameDimensionality(data, centroids, "KMeans::Cluster()");
  }

  // Use the partitioner to come up with the partition assignments and calculate
  // the initial centroids.
  if (!initialGuess)
  {
    // The GetInitialAssignmentsOrCentroids() function will call the appropriate
    // function in the InitialPartitionPolicy to return either assignments or
    // centroids.  We prefer centroids, but if assignments are returned, then we
    // have to calculate the initial centroids for the first iteration.
    arma::Row<size_t> assignments;
    bool gotAssignments = GetInitialAssignmentsOrCentroids(partitioner, data,
        clusters, assignments, centroids);
    if (gotAssignments)
    {
      // The partitioner gives assignments, so we need to calculate centroids
      // from those assignments.
      arma::Row<size_t> counts;
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
  }

  // Counts of points in each cluster.
  arma::Col<size_t> counts(clusters);

  size_t iteration = 0;

  LloydStepType<DistanceType, MatType> lloydStep(data, distance);
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
    for (size_t i = 0; i < counts.n_elem; ++i)
    {
      if (counts[i] == 0)
      {
        Log::Info << "Cluster " << i << " is empty.\n";
        if (iteration % 2 == 0)
          emptyClusterAction.EmptyCluster(data, i, centroids, centroidsOther,
              counts, distance, iteration);
        else
          emptyClusterAction.EmptyCluster(data, i, centroidsOther, centroids,
              counts, distance, iteration);
      }
    }

    iteration++;
    Log::Info << "KMeans::Cluster(): iteration " << iteration << ", residual "
        << cNorm << ".\n";
    if (std::isnan(cNorm) || std::isinf(cNorm))
      cNorm = 1e-4; // Keep iterating.
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
template<typename DistanceType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
void KMeans<
    DistanceType,
    InitialPartitionPolicy,
    EmptyClusterPolicy,
    LloydStepType,
    MatType>::
Cluster(const MatType& data,
        const size_t clusters,
        arma::Row<size_t>& assignments,
        arma::mat& centroids,
        const bool initialAssignmentGuess,
        const bool initialCentroidGuess)
{
  // Now, the initial assignments.  First determine if they are necessary.
  if (initialAssignmentGuess)
  {
    util::CheckSameSizes(data, assignments, "KMeans::Cluster()", "assignments");

    // Calculate initial centroids.
    arma::Row<size_t> counts;
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

  // Calculate final assignments in parallel over the entire dataset.
  assignments.set_size(data.n_cols);

  #pragma omp parallel for
  for (size_t i = 0; i < (size_t) data.n_cols; ++i)
  {
    // Find the closest centroid to this point.
    double minDistance = std::numeric_limits<double>::infinity();
    size_t closestCluster = centroids.n_cols; // Invalid value.

    for (size_t j = 0; j < centroids.n_cols; ++j)
    {
      const double dist = distance.Evaluate(data.col(i), centroids.col(j));

      if (dist < minDistance)
      {
        minDistance = dist;
        closestCluster = j;
      }
    }

    Log::Assert(closestCluster != centroids.n_cols);
    assignments[i] = closestCluster;
  }
}

template<typename DistanceType,
         typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType,
         typename MatType>
template<typename Archive>
void KMeans<DistanceType,
            InitialPartitionPolicy,
            EmptyClusterPolicy,
            LloydStepType,
            MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(maxIterations));
  ar(CEREAL_NVP(distance));
  ar(CEREAL_NVP(partitioner));
  ar(CEREAL_NVP(emptyClusterAction));
}

} // namespace mlpack
