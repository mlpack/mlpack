/**
 * @file mean_shift_impl.hpp
 * @author Shangtong Zhang
 *
 * Mean shift clustering implementation.
 */
#ifndef __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_IMPL_HPP
#define __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_IMPL_HPP

#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search_stat.hpp>

// In case it hasn't been included yet.
#include "mean_shift.hpp"

namespace mlpack {
namespace meanshift {

/**
  * Construct the Mean Shift object.
  */
template<typename KernelType, typename MatType>
MeanShift<KernelType, MatType>::MeanShift(const double radius,
                                          const size_t maxIterations,
                                          const KernelType kernel) :
    maxIterations(maxIterations),
    kernel(kernel),
{
  // Set the radius; estimate if needed.
  Radius(radius);
}

template<typename KernelType, typename MatType>
void MeanShift<KernelType, MatType>::Radius(double radius)
{
  this->radius = radius;
}

// Estimate radius based on given dataset.
template<typename KernelType, typename MatType>
double MeanShift<KernelType, MatType>::EstimateRadius(const MatType& data)
{
  neighbor::NeighborSearch<
      neighbor::NearestNeighborSort,
      metric::EuclideanDistance,
      tree::BinarySpaceTree<bound::HRectBound<2>,
            neighbor::NeighborSearchStat<neighbor::NearestNeighborSort> >
  > neighborSearch(data);

  /**
   * For each point in dataset, select nNeighbors nearest points and get
   * nNeighbors distances.  Use the maximum distance to estimate the duplicate
   * threshhold.
   */
  size_t nNeighbors = size_t(data.n_cols * 0.3);
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighborSearch.Search(nNeighbors, neighbors, distances);

  // Get max distance for each point.
  arma::rowvec maxDistances = max(distances);

  // Calculate and return the radius.
  return sum(maxDistances) / (double) data.n_cols;
}

// General way to calculate the weight of a data point.
template<typename KernelType, typename MatType>
bool MeanShift<KernelType, MatType>::CalcWeight(
    const arma::colvec& centroid,
    const arma::colvec& point,
    double& weight)
{
  double distance = EuclideanDistance::Evaluate(centroid, point);
  if (distance >= radius || distance == 0)
    return false;

  distance /= radius;
  weight = kernel.Gradient(distance) / distance;
  return true;
}

/**
 * Perform Mean Shift clustering on the data, returning a list of cluster
 * assignments and centroids.
 */
template<typename KernelType, typename MatType>
inline void MeanShift<KernelType, MatType>::Cluster(
    const MatType& data,
    arma::Col<size_t>& assignments,
    arma::mat& centroids)
{
  if (radius <= 0)
  {
    // An invalid radius is given; an estimation is needed.
    Radius(EstimateRadius(data));
  }

  // Holds all centroids, before removing duplicate ones.
  arma::mat allCentroids(data.n_rows, data.n_cols);
  assignments.set_size(data.n_cols);

  // For each point in dataset, perform mean shift algorithm.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Initial centroid is the point itself.
    allCentroids.col(i) = data.col(i);

    for (size_t completedIterations = 0; completedIterations < maxIterations;
         completedIterations++)
    {
      // Store new centroid in this.
      arma::Col<double> newCentroid = arma::zeros(data.n_rows, 1);

      double sumWeight = 0;

      // Go through all the points
      for (size_t j = 0; j < data.n_cols; ++j)
      {
        double weight = 0;
        if (CalcWeight(allCentroids.col(i), data.col(j), weight))
        {
          sumWeight += weight;
          newCentroid += weight * data.col(j);
        }
      }

      newCentroid /= sumWeight;

      // Calculate the mean shift vector.
      arma::Col<double> mhVector = newCentroid - allCentroids.col(i);

      // If the mean shift vector is small enough, it has converged.
      if (EuclideanDistance::Evaluate(newCentroid, allCentroids.col(i)) <
          1e-3 * radius)
      {
        // Determine if the new centroid is duplicate with old ones.
        bool isDuplicated = false;
        for (size_t k = 0; k < centroids.n_cols; ++k)
        {
          arma::Col<double> delta = allCentroids.col(i) - centroids.col(k);
          if (norm(delta, 2) < radius)
          {
            isDuplicated = true;
            assignments(i) = k;
            break;
          }
        }

        if (!isDuplicated)
        {
          // This centroid is a new centroid.
          if (centroids.n_cols == 0)
          {
            centroids.insert_cols(0, allCentroids.col(i));
            assignments(i) = 0;
          }
          else
          {
            centroids.insert_cols(centroids.n_cols, allCentroids.col(i));
            assignments(i) = centroids.n_cols - 1;
          }
        }

        // Get out of the loop.
        break;
      }

      // Update the centroid.
      allCentroids.col(i) = newCentroid;
    }
  }
}

} // namespace meanshift
} // namespace mlpack

#endif
