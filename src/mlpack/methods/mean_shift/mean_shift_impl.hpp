/**
 * @file mean_shift_impl.hpp
 * @author Shangtong Zhang
 *
 * Mean Shift clustering
 */

#include "mean_shift.hpp"
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search_stat.hpp>



namespace mlpack {
namespace meanshift {
  
/**
  * Construct the Mean Shift object.
  */
template<typename KernelType,
         typename MatType>
MeanShift<
  KernelType,
  MatType>::
MeanShift(const double radius,
          const size_t maxIterations,
          const KernelType kernel) :
    maxIterations(maxIterations),
    kernel(kernel)
{
  Radius(radius);
}
  
template<typename KernelType,
         typename MatType>
void MeanShift<
  KernelType,
  MatType>::
Radius(double radius) {
  this->radius = radius;
  squaredRadius = radius * radius;
}
  
// Estimate radius based on given dataset.
template<typename KernelType,
         typename MatType>
double MeanShift<
  KernelType,
  MatType>::
estimateRadius(const MatType &data) {
  
  neighbor::NeighborSearch<
    neighbor::NearestNeighborSort,
    metric::EuclideanDistance,
    tree::BinarySpaceTree<bound::HRectBound<2>,
          neighbor::NeighborSearchStat<neighbor::NearestNeighborSort> >
    > neighborSearch(data);
  
  /** 
   * For each point in dataset, 
   * select nNeighbors nearest points and get nNeighbors distances. 
   * Use the maximum distance to estimate the duplicate thresh.
   */
  size_t nNeighbors = (int)(data.n_cols * 0.3);
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighborSearch.Search(nNeighbors, neighbors, distances);
  
  // Get max distance for each point.
  arma::rowvec maxDistances = max(distances);
  
  // Calc and return the duplicate thresh.
  return sum(maxDistances) / (double)data.n_cols;
  
}

/**
 * Perform Mean Shift clustering on the data, returning a list of cluster
 * assignments and centroids.
 */
template<typename KernelType,
         typename MatType>
inline void MeanShift<
    KernelType,
    MatType>::
Cluster(const MatType& data,
        arma::Col<size_t>& assignments,
        arma::mat& centroids) {
  
  if (radius <= 0) {
    // An invalid radius is given, an estimation is needed.
    Radius(estimateRadius(data));
  }
  
  // all centroids before remove duplicate ones.
  arma::mat allCentroids(data.n_rows, data.n_cols);
  assignments.set_size(data.n_cols);
  
  // for each point in dataset, perform mean shift algorithm.
  for (size_t i = 0; i < data.n_cols; ++i) {
    
    //initial centroid is the point itself.
    allCentroids.col(i) = data.col(i);
    
    for (size_t completedIterations = 0; completedIterations < maxIterations;
         completedIterations++) {
      
      // to store new centroid
      arma::Col<double> newCentroid = arma::zeros(data.n_rows, 1);
      
      double sumWeight = 0;
      
      // Go through all the points
      for (size_t j = 0; j < data.n_cols; ++j) {
        
        // Calculate the distance between old centroid and current point.
        double squaredDist = metric::SquaredEuclideanDistance::
                            Evaluate(allCentroids.col(i), data.col(j));
        
        // If current point is near the old centroid
        if (squaredDist < squaredRadius) {
          
          // calculate weight for current point
          double weight = kernel.Gradient(squaredDist / squaredRadius);
          
          sumWeight += weight;
          
          // update new centroid.
          newCentroid += weight * data.col(j);
          
        }
        
      }
      
      newCentroid /= sumWeight;
      
      // calc the mean shift vector.
      arma::Col<double> mhVector = newCentroid - allCentroids.col(i);
      
      // update the centroid.
      allCentroids.col(i) = newCentroid;
      
      // If the 2-norm of mean shift vector is small enough, it has converged.
      if (arma::norm(mhVector, 2) < 1e-3 * radius) {
        
        // Determine if the new centroid is duplicate with old ones.
        bool isDuplicated = false;
        for (size_t k = 0; k < centroids.n_cols; ++k) {
          arma::Col<double> delta = allCentroids.col(i) - centroids.col(k);
          if (norm(delta, 2) < radius) {
            isDuplicated = true;
            assignments(i) = k;
            break;
          }
        }
        
        if (!isDuplicated) {
          // this centroid is a new centroid.
          if (centroids.n_cols == 0) {
            centroids.insert_cols(0, allCentroids.col(i));
            assignments(i) = 0;
          } else {
            centroids.insert_cols(centroids.n_cols, allCentroids.col(i));
            assignments(i) = centroids.n_cols - 1;
          }
        }
        
        // Get out of the loop.
        break;
      }
      
    }
    
  }
  
}
  
}; // namespace meanshift
}; // namespace mlpack