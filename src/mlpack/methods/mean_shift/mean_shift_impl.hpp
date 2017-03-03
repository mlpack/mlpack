/**
 * @file mean_shift_impl.hpp
 * @author Shangtong Zhang
 *
 * Mean shift clustering implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_IMPL_HPP
#define MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_IMPL_HPP

#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/range_search/range_search.hpp>

#include "map"

// In case it hasn't been included yet.
#include "mean_shift.hpp"

namespace mlpack {
namespace meanshift {

/**
  * Construct the Mean Shift object.
  */
template<bool UseKernel, typename KernelType, typename MatType>
MeanShift<UseKernel, KernelType, MatType>::
MeanShift(const double radius,
          const size_t maxIterations,
          const KernelType kernel) :
    radius(radius),
    maxIterations(maxIterations),
    kernel(kernel)
{
  // Nothing to do.
}

template<bool UseKernel, typename KernelType, typename MatType>
void MeanShift<UseKernel, KernelType, MatType>::Radius(double radius)
{
  this->radius = radius;
}

// Estimate radius based on given dataset.
template<bool UseKernel, typename KernelType, typename MatType>
double MeanShift<UseKernel, KernelType, MatType>::
EstimateRadius(const MatType& data, double ratio)
{
  neighbor::KNN neighborSearch(data);

  /**
   * For each point in dataset, select nNeighbors nearest points and get
   * nNeighbors distances.  Use the maximum distance to estimate the duplicate
   * threshhold.
   */
  const size_t nNeighbors = size_t(data.n_cols * ratio);
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighborSearch.Search(nNeighbors, neighbors, distances);

  // Get max distance for each point.
  arma::rowvec maxDistances = max(distances);

  // Calculate and return the radius.
  return sum(maxDistances) / (double) data.n_cols;
}

// Class to compare two vectors.
template <typename VecType>
class less
{
 public:
  bool operator()(const VecType& first, const VecType& second) const
  {
    for (size_t i = 0; i < first.n_rows; ++i)
    {
      if (first[i] == second[i])
        continue;
      return first(i) < second(i);
    }
    return false;
  }
};

// Generate seeds from given data set.
template<bool UseKernel, typename KernelType, typename MatType>
void MeanShift<UseKernel, KernelType, MatType>::GenSeeds(
    const MatType& data,
    const double binSize,
    const int minFreq,
    MatType& seeds)
{
  typedef arma::colvec VecType;
  std::map<VecType, int, less<VecType> > allSeeds;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    VecType binnedPoint = arma::floor(data.unsafe_col(i) / binSize);
    if (allSeeds.find(binnedPoint) == allSeeds.end())
      allSeeds[binnedPoint] = 1;
    else
      allSeeds[binnedPoint]++;
  }

  // Remove seeds with too few points.  First we count the number of seeds we
  // end up with, then we add them.
  std::map<VecType, int, less<VecType> >::iterator it;
  size_t count = 0;
  for (it = allSeeds.begin(); it != allSeeds.end(); ++it)
    if (it->second >= minFreq)
      ++count;

  seeds.set_size(data.n_rows, count);
  count = 0;
  for (it = allSeeds.begin(); it != allSeeds.end(); ++it)
  {
    if (it->second >= minFreq)
    {
      seeds.col(count) = it->first;
      ++count;
    }
  }

  seeds *= binSize;
}

// Calculate new centroid with given kernel.
template<bool UseKernel, typename KernelType, typename MatType>
template<bool ApplyKernel>
typename std::enable_if<ApplyKernel, bool>::type
MeanShift<UseKernel, KernelType, MatType>::
CalculateCentroid(const MatType& data,
                  const std::vector<size_t>& neighbors,
                  const std::vector<double>& distances,
                  arma::colvec& centroid)
{
  double sumWeight = 0;
  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    if (distances[i] > 0)
    {
      double dist = distances[i] / radius;
      double weight = kernel.Gradient(dist) / dist;
      sumWeight += weight;
      centroid += weight * data.unsafe_col(neighbors[i]);
    }
  }

  if (sumWeight != 0)
  {
    centroid /= sumWeight;
    return true;
  }
  return false;
}

// Calculate new centroid by mean.
template<bool UseKernel, typename KernelType, typename MatType>
template<bool ApplyKernel>
typename std::enable_if<!ApplyKernel, bool>::type
MeanShift<UseKernel, KernelType, MatType>::
CalculateCentroid(const MatType& data,
                  const std::vector<size_t>& neighbors,
                  const std::vector<double>&, /*unused*/
                  arma::colvec& centroid)
{
  for (size_t i = 0; i < neighbors.size(); ++i)
    centroid += data.unsafe_col(neighbors[i]);

  centroid /= neighbors.size();
  return true;
}

/**
 * Perform Mean Shift clustering on the data set, returning a list of cluster
 * assignments and centroids.
 */
template<bool UseKernel, typename KernelType, typename MatType>
inline void MeanShift<UseKernel, KernelType, MatType>::Cluster(
    const MatType& data,
    arma::Col<size_t>& assignments,
    arma::mat& centroids,
    bool useSeeds)
{
  if (radius <= 0)
  {
    // An invalid radius is given; an estimation is needed.
    Radius(EstimateRadius(data));
  }

  MatType seeds;
  const MatType* pSeeds = &data;
  if (useSeeds)
  {
    GenSeeds(data, radius, 1, seeds);
    pSeeds = &seeds;
  }

  // Holds all centroids before removing duplicate ones.
  arma::mat allCentroids(pSeeds->n_rows, pSeeds->n_cols);

  assignments.set_size(data.n_cols);

  range::RangeSearch<> rangeSearcher(data);
  math::Range validRadius(0, radius);
  std::vector<std::vector<size_t> > neighbors;
  std::vector<std::vector<double> > distances;

  // For each seed, perform mean shift algorithm.
  for (size_t i = 0; i < pSeeds->n_cols; ++i)
  {
    // Initial centroid is the seed itself.
    allCentroids.col(i) = pSeeds->unsafe_col(i);
    for (size_t completedIterations = 0; completedIterations < maxIterations;
         completedIterations++)
    {
      // Store new centroid in this.
      arma::colvec newCentroid = arma::zeros<arma::colvec>(pSeeds->n_rows);

      rangeSearcher.Search(allCentroids.unsafe_col(i), validRadius,
          neighbors, distances);
      if (neighbors[0].size() <= 1)
        break;

      // Calculate new centroid.
      if (!CalculateCentroid(data, neighbors[0], distances[0], newCentroid))
        newCentroid = allCentroids.unsafe_col(i);

      // If the mean shift vector is small enough, it has converged.
      if (metric::EuclideanDistance::Evaluate(newCentroid,
          allCentroids.unsafe_col(i)) < 1e-3 * radius)
      {
        // Determine if the new centroid is duplicate with old ones.
        bool isDuplicated = false;
        for (size_t k = 0; k < centroids.n_cols; ++k)
        {
          const double distance = metric::EuclideanDistance::Evaluate(
              allCentroids.unsafe_col(i), centroids.unsafe_col(k));
          if (distance < radius)
          {
            isDuplicated = true;
            break;
          }
        }

        if (!isDuplicated)
          centroids.insert_cols(centroids.n_cols, allCentroids.unsafe_col(i));

        // Get out of the loop.
        break;
      }

      // Update the centroid.
      allCentroids.col(i) = newCentroid;
    }
  }

  // Assign centroids to each point.
  neighbor::KNN neighborSearcher(centroids);
  arma::mat neighborDistances;
  arma::Mat<size_t> resultingNeighbors;
  neighborSearcher.Search(data, 1, resultingNeighbors, neighborDistances);
  assignments = resultingNeighbors.t();
}

} // namespace meanshift
} // namespace mlpack

#endif
