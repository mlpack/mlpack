/**
 * @file methods/mean_shift/mean_shift_impl.hpp
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
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/range_search/range_search.hpp>

#include "map"

// In case it hasn't been included yet.
#include "mean_shift.hpp"

namespace mlpack {

/**
 * Construct the Mean Shift object.
 */
template<bool UseKernel, typename KernelType>
MeanShift<UseKernel, KernelType>::MeanShift(const double radius,
                                            const size_t maxIterations,
                                            const KernelType kernel) :
    radius(radius),
    maxIterations(maxIterations),
    kernel(kernel)
{
  // Nothing to do.
}

template<bool UseKernel, typename KernelType>
void MeanShift<UseKernel, KernelType>::Radius(double radius)
{
  this->radius = radius;
}

// Estimate radius based on given dataset.
template<bool UseKernel, typename KernelType>
template<typename MatType>
typename MatType::elem_type
MeanShift<UseKernel, KernelType>::EstimateRadius(const MatType& data,
                                                 double ratio)
{
  NeighborSearch<NearestNeighborSort, EuclideanDistance, MatType>
      neighborSearch(data);

  /**
   * For each point in dataset, select nNeighbors nearest points and get
   * nNeighbors distances.  Use the maximum distance to estimate the duplicate
   * threshhold.
   */
  const size_t nNeighbors = size_t(data.n_cols * ratio);
  arma::Mat<size_t> neighbors;
  MatType distances;
  neighborSearch.Search(nNeighbors, neighbors, distances);

  // Calculate and return the radius.
  return sum(max(distances)) / (typename MatType::elem_type) data.n_cols;
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
template<bool UseKernel, typename KernelType>
template<typename MatType, typename CentroidsType>
void MeanShift<UseKernel, KernelType>::GenSeeds(const MatType& data,
                                                const double binSize,
                                                const int minFreq,
                                                CentroidsType& seeds)
{
  using VecType = typename GetColType<MatType>::type;
  using CentroidVecType = typename GetColType<CentroidsType>::type;
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
  typename std::map<VecType, int, less<VecType> >::iterator it;
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
      seeds.col(count) = arma::conv_to<CentroidVecType>::from(it->first);
      ++count;
    }
  }

  seeds *= binSize;
}

// Calculate new centroid with given kernel.
template<bool UseKernel, typename KernelType>
template<bool ApplyKernel, typename MatType, typename VecType>
std::enable_if_t<ApplyKernel, bool>
MeanShift<UseKernel, KernelType>::CalculateCentroid(
    const MatType& data,
    const std::vector<size_t>& neighbors,
    const std::vector<typename MatType::elem_type>& distances,
    VecType& centroid)
{
  using ElemType = typename MatType::elem_type;

  ElemType sumWeight = 0;
  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    if (distances[i] > 0)
    {
      ElemType dist = distances[i] / radius;
      ElemType weight = kernel.Gradient(dist) / dist;
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
template<bool UseKernel, typename KernelType>
template<bool ApplyKernel, typename MatType, typename VecType>
std::enable_if_t<!ApplyKernel, bool>
MeanShift<UseKernel, KernelType>::CalculateCentroid(
    const MatType& data,
    const std::vector<size_t>& neighbors,
    const std::vector<typename MatType::elem_type>&, /*unused*/
    VecType& centroid)
{
  for (size_t i = 0; i < neighbors.size(); ++i)
    centroid += data.unsafe_col(neighbors[i]);

  centroid /= neighbors.size();
  return true;
}

/**
 * Perform Mean Shift clustering on the data set, returning a list of centroids.
 */
template<bool UseKernel, typename KernelType>
template<typename MatType, typename CentroidsType>
inline void MeanShift<UseKernel, KernelType>::Cluster(
    const MatType& data,
    CentroidsType& centroids,
    bool forceConvergence,
    bool useSeeds)
{
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;
  using VecType = typename GetColType<MatType>::type;

  if (radius <= 0)
  {
    // An invalid radius is given; an estimation is needed.
    Radius(EstimateRadius(data));
  }

  CentroidsType seeds;
  const MatType* pSeeds = &data;
  if (useSeeds)
  {
    GenSeeds(data, radius, 1, seeds);
    pSeeds = &seeds;
  }

  // Holds all centroids before removing duplicate ones.
  CentroidsType allCentroids(pSeeds->n_rows, pSeeds->n_cols);

  RangeSearch<EuclideanDistance, MatType> rangeSearcher(data);
  RangeType<ElemType> validRadius((ElemType) 0, (ElemType) radius);
  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<ElemType>> distances;

  // For each seed, perform mean shift algorithm.
  for (size_t i = 0; i < pSeeds->n_cols; ++i)
  {
    // Initial centroid is the seed itself.
    allCentroids.col(i) = pSeeds->unsafe_col(i);
    for (size_t completedIterations = 0; completedIterations < maxIterations
        || forceConvergence; completedIterations++)
    {
      // Store new centroid in this.
      VecType newCentroid = zeros<VecType>(pSeeds->n_rows);

      rangeSearcher.Search(allCentroids.unsafe_col(i), validRadius,
          neighbors, distances);
      if (neighbors[0].size() == 0) // There are no points in the cluster.
        break;

      // Calculate new centroid.
      if (!CalculateCentroid(data, neighbors[0], distances[0], newCentroid))
        newCentroid = allCentroids.unsafe_col(i);

      // If the mean shift vector is small enough, it has converged.
      if (EuclideanDistance::Evaluate(newCentroid,
          allCentroids.unsafe_col(i)) < 1e-3 * radius)
      {
        // Determine if the new centroid is duplicate with old ones.
        bool isDuplicated = false;
        for (size_t k = 0; k < centroids.n_cols; ++k)
        {
          const ElemType distance = EuclideanDistance::Evaluate(
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

  // If no centroid has converged due to too little iterations and without
  // forcing convergence, take 1 random centroid calculated.
  if (centroids.empty())
  {
    Log::Warn << "No clusters converged; setting 1 random centroid calculated. "
        << "Try increasing the maximum number of iterations or setting the "
        << "option to force convergence." << std::endl;

    if (maxIterations == 0)
    {
      centroids.insert_cols(centroids.n_cols, data.col(0));
    }
    else
    {
      centroids.insert_cols(centroids.n_cols, allCentroids.col(0));
    }
  }
}

/**
 * Perform Mean Shift clustering on the data set, returning a list of cluster
 * assignments and centroids.
 */
template<bool UseKernel, typename KernelType>
template<typename MatType, typename LabelsType, typename CentroidsType>
inline void MeanShift<UseKernel, KernelType>::Cluster(
    const MatType& data,
    LabelsType& assignments,
    CentroidsType& centroids,
    bool forceConvergence,
    bool useSeeds)
{
  // Perform the actual clustering.
  Cluster(data, centroids, forceConvergence, useSeeds);

  assignments.set_size(data.n_cols);
  if (centroids.n_cols == 1)
  {
    assignments.zeros();
  }
  else
  {
    // Assign centroids to each point.
    //
    // NeighborSearch only supports when the reference and query set have the
    // same type, so forcibly convert the centroids to the same type as data if
    // needed.  This also means we have to separate out the neighbor searching
    // operation to a utility function, so that the compiler doesn't try to
    // instantiate the NeighborSearch class with invalid types.
    arma::Mat<typename MatType::elem_type> neighborDistances;
    arma::Mat<size_t> resultingNeighbors;
    NeighborSearch<NearestNeighborSort, EuclideanDistance, MatType>
        neighborSearcher(centroids);
    neighborSearcher.Search(data, 1, resultingNeighbors, neighborDistances);
    assignments = resultingNeighbors;
  }
}

} // namespace mlpack

#endif
