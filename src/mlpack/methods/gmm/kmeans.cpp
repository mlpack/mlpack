/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file kmeans.cpp
 *
 * Implementation for the K-means method for getting an initial point.
 *
 */
#include "kmeans.hpp"

#include <mlpack/core/kernels/lmetric.hpp>

namespace mlpack {
namespace gmm {

void KMeans(const arma::mat& data,
            const size_t value_of_k,
            std::vector<arma::vec>& means,
            std::vector<arma::mat>& covars,
            arma::vec& weights) {
  // Make sure we have more points than clusters.
  if (value_of_k > data.n_cols)
    Log::Warn << "k-means: more clusters requested than points given.  Empty"
        << " clusters may result." << std::endl;

  // Assignment of cluster of each point.
  arma::Col<size_t> assignments(data.n_cols); // Col used so we have shuffle().
  // Centroids of each cluster.  Each column corresponds to a centroid.
  arma::mat centroids(data.n_rows, value_of_k);
  // Counts of points in each cluster.
  arma::Col<size_t> counts(value_of_k);

  // First we must randomly partition the dataset.
  assignments = arma::shuffle(arma::linspace<arma::Col<size_t> >(0,
      value_of_k - 1, data.n_cols));

  // Set counts correctly.
  for (size_t i = 0; i < value_of_k; i++)
    counts[i] = accu(assignments == i);

  size_t changed_assignments = 0;
  do
  {
    // Update step.
    // Calculate centroids based on given assignments.
    centroids.zeros();

    for (size_t i = 0; i < data.n_cols; i++)
      centroids.col(assignments[i]) += data.col(i);

    for (size_t i = 0; i < value_of_k; i++)
      centroids.col(i) /= counts[i];

    // Assignment step.
    // Find the closest centroid to each point.  We will keep track of how many
    // assignments change.  When no assignments change, we are done.
    changed_assignments = 0;
    for (size_t i = 0; i < data.n_cols; i++)
    {
      // Find the closest centroid to this point.
      double min_distance = std::numeric_limits<double>::infinity();
      size_t closest_cluster = value_of_k; // Invalid value.

      for (size_t j = 0; j < value_of_k; j++)
      {
        double distance = kernel::SquaredEuclideanDistance::Evaluate(
            data.unsafe_col(i), centroids.unsafe_col(j));

        if (distance < min_distance)
        {
          min_distance = distance;
          closest_cluster = j;
        }
      }

      // Reassign this point to the closest cluster.
      if (assignments[i] != closest_cluster)
      {
        // Update counts.
        counts[assignments[i]]--;
        counts[closest_cluster]++;
        // Update assignment.
        assignments[i] = closest_cluster;
        changed_assignments++;
      }
    }

    // Keep-bad-things-from-happening step.
    // Ensure that no cluster is empty, and if so, take corrective action.
    for (size_t i = 0; i < value_of_k; i++)
    {
      if (counts[i] == 0)
      {
        // Strategy: take the furthest point from the cluster with highest
        // variance.  So, we need the variance of each cluster.
        arma::vec variances;
        variances.zeros(value_of_k);
        for (size_t j = 0; j < data.n_cols; j++)
          variances[assignments[j]] += var(data.col(j));

        size_t cluster;
        double max_var = 0;
        for (size_t j = 0; j < value_of_k; j++)
        {
          if (variances[j] > max_var)
          {
            cluster = j;
            max_var = variances[j];
          }
        }

        // Now find the furthest point.
        size_t point = data.n_cols; // Invalid.
        double distance = 0;
        for (size_t j = 0; j < data.n_cols; j++)
        {
          if (assignments[j] == cluster)
          {
            double d = kernel::SquaredEuclideanDistance::Evaluate(
                data.unsafe_col(j), centroids.unsafe_col(cluster));

            if (d >= distance)
            {
              distance = d;
              point = j;
            }
          }
        }

        // Take that point and add it to the empty cluster.
        counts[cluster]--;
        counts[i]++;
        assignments[point] = i;
        changed_assignments++;
      }
    }

  } while (changed_assignments > 0);

  // Now, with the centroids final, we need to find the covariance matrix of
  // each cluster and then the a priori weight.  We also need to assign the
  // means to be the centroids.  First, we must make sure the size of the
  // vectors is correct.
  means.resize(value_of_k);
  covars.resize(value_of_k);
  weights.set_size(value_of_k);
  for (size_t i = 0; i < value_of_k; i++)
  {
    // Assign mean.
    means[i] = centroids.col(i);

    // Calculate covariance.
    arma::mat data_subset(data.n_rows, accu(assignments == i));
    size_t position = 0;
    for (size_t j = 0; j < data.n_cols; j++)
    {
      if (assignments[j] == i)
      {
        data_subset.col(position) = data.col(j);
        position++;
      }
    }

    covars[i] = ccov(data_subset);

    // Assign weight.
    weights[i] = (double) accu(assignments == i) / (double) data.n_cols;
  }
}

}; // namespace gmm
}; // namespace mlpack
