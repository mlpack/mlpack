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
#include <mlpack/methods/range_search/range_search.hpp>

#include "map"

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
    radius(radius),
    maxIterations(maxIterations),
    kernel(kernel)
{
  // Nothing to do.
}

template<typename KernelType, typename MatType>
void MeanShift<KernelType, MatType>::Radius(double radius)
{
  this->radius = radius;
}

// Estimate radius based on given dataset.
template<typename KernelType, typename MatType>
double MeanShift<KernelType, MatType>::EstimateRadius(const MatType& data,
                                                      double ratio)
{
  neighbor::AllkNN neighborSearch(data);
  /**
   * For each point in dataset, select nNeighbors nearest points and get
   * nNeighbors distances.  Use the maximum distance to estimate the duplicate
   * threshhold.
   */
  size_t nNeighbors = size_t(data.n_cols * ratio);
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighborSearch.Search(nNeighbors, neighbors, distances);

  // Get max distance for each point.
  arma::rowvec maxDistances = max(distances);

  // Calculate and return the radius.
  return sum(maxDistances) / (double) data.n_cols;
}

// Class to compare two vector
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

// Generate seeds form given data set
template<typename KernelType, typename MatType>
void MeanShift<KernelType, MatType>::genSeeds(
    const MatType& data,
    double binSize,
    int minFreq,
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
  
  // Remove seeds with too few points
  std::map<VecType, int, less<VecType> >::iterator it;
  for (it = allSeeds.begin(); it != allSeeds.end(); ++it)
  {
    if (it->second >= minFreq)
      seeds.insert_cols(seeds.n_cols, it->first);
  }
  
  seeds = seeds * binSize;
}

/**
 * Perform Mean Shift clustering on the data set, returning a list of cluster
 * assignments and centroids.
 */
template<typename KernelType, typename MatType>
inline void MeanShift<KernelType, MatType>::Cluster(
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
    genSeeds(data, radius, 1, seeds);
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
      arma::colvec newCentroid(pSeeds->n_rows, arma::fill::zeros);
      
      double sumWeight = 0;
      rangeSearcher.Search(allCentroids.unsafe_col(i), validRadius,
          neighbors, distances);
      if (neighbors[0].size() <= 1)
        break;
      for (size_t j = 0; j < neighbors[0].size(); ++j)
      {
        if (distances[0][j] > 0)
        {
          distances[0][j] /= radius;
          double weight = kernel.Gradient(distances[0][j]) / distances[0][j];
          sumWeight += weight;
          newCentroid += weight * data.unsafe_col(neighbors[0][j]);
        }
      }
      
      if (sumWeight != 0)
        newCentroid /= sumWeight;
      else
        newCentroid = allCentroids.unsafe_col(i);

      // If the mean shift vector is small enough, it has converged.
      if (metric::EuclideanDistance::Evaluate(newCentroid, allCentroids.unsafe_col(i)) <
          1e-3 * radius)
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
  
  // Assign centroids to each point
  neighbor::AllkNN neighborSearcher(centroids, data);
  arma::mat neighborDistances;
  arma::Mat<size_t> resultingNeighbors;
  neighborSearcher.Search(1, resultingNeighbors, neighborDistances);
  assignments = resultingNeighbors.t();
}

} // namespace meanshift
} // namespace mlpack

#endif
