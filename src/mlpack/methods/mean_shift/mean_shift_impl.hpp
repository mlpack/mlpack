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
MeanShift(const double duplicateThresh,
          const size_t maxIterations,
          const double stopThresh,
          const KernelType kernel) :
    duplicateThresh(duplicateThresh),
    maxIterations(maxIterations),
    stopThresh(stopThresh),
    kernel(kernel)
{
  // Nothing to do.
}
  
  
// Estimate duplicate thresh based on given dataset.
template<typename KernelType,
         typename MatType>
double MeanShift<
  KernelType,
  MatType>::
estimateDuplicateThresh(const MatType &data) {
  
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
  
  if (duplicateThresh < 0) {
    // An invalid duplicate thresh is given, an estimation is needed.
    duplicateThresh = estimateDuplicateThresh(data);
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
      
      // new centroid
      arma::Col<double> newCentroid = arma::zeros(data.n_rows, 1);
      
      double sumWeight = 0;
      for (size_t j = 0; j < data.n_cols; ++j) {
        
        // calc weight for each point
        double weight = kernel.Evaluate(allCentroids.col(i), data.col(j));
        sumWeight += weight;
        
        // update new centroid.
        newCentroid += weight * data.col(j);
        
      }
      
      newCentroid /= sumWeight;
      
      completedIterations ++;
      
      // calc the mean shift vector.
      arma::Col<double> mhVector = newCentroid - allCentroids.col(i);
      
      // update the centroid.
      allCentroids.col(i) = newCentroid;
      
      if (arma::norm(mhVector, 2) < stopThresh) {
        break;
      }
      
    }
    
  }
  
  // remove duplicate centroids.
  for (size_t i = 0; i < allCentroids.n_cols; ++i) {
    
    bool isDuplicated = false;
    
    /** 
     * if a centroid is a neighbouring point of existing points,
     * remove it and update corresponding assignments.
     */
    for (size_t j = 0; j < centroids.n_cols; ++j) {
      arma::Col<double> delta = allCentroids.col(i) - centroids.col(j);
      if (norm(delta, 2) < duplicateThresh) {
        isDuplicated = true;
        assignments(i) = j;
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
  }
  
}
  
}; // namespace meanshift
}; // namespace mlpack