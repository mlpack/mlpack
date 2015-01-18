/**
 * @file mean_shift.hpp
 * @author Shangtong Zhang
 *
 * Mean Shift clustering
 */

#ifndef __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP
#define __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace meanshift /** Mean Shift clustering. */ {

/**
 * This class implements Mean Shift clustering.
 * For each point in dataset, apply mean shift algorithm until maximum 
 * iterations or convergence. 
 * Then remove duplicate centroids.
 * 
 * A simple example of how to run Mean Shift clustering is shown below.
 *
 * @code
 * extern arma::mat data; // Dataset we want to run Mean Shift on.
 * arma::Col<size_t> assignments; // Cluster assignments.
 * arma::mat centroids; // Cluster centroids.
 * extern int maxIterations; // Maximum number of iterations.
 * extern double stopThresh; //
 * extern double radius; //
 *
 * MeanShift<arma::mat, metric::EuclideanDistance> meanShift(maxIterations, 
 *          stopThresh, radius);
 * meanShift.Cluster(dataset, assignments, centroids);
 * @endcode
 *
 * @tparam MetricType The distance metric to use for this KMeans; see
 *     metric::LMetric for an example.
 */
  
template<typename MatType = arma::mat,
         typename MetricType = metric::EuclideanDistance>
class MeanShift
{
 public:
  /**
   * Create a Mean Shift object and set the parameters which Mean Shift
   * will be run with.
   *
   * @param maxIterations Maximum number of iterations allowed before giving up
   * @param stopThresh If the 2-norm of the mean shift vector is less than stopThresh, 
   *     iterations will terminate.
   * @param radius When iterating, take points within distance of
   *     radius into consideration and two centroids within distance
   *     of radius will be ragarded as one centroid.
   * @param metric Optional MetricType object; for when the metric has state
   *     it needs to store.
   */
  MeanShift(const size_t maxIterations,
            const double stopThresh,
            const double radius,
            const MetricType metric = MetricType());
  
  
  /**
   * Perform Mean Shift clustering on the data, returning a list of cluster
   * assignments and centroids.
   * 
   * @tparam MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param assignments Vector to store cluster assignments in.
   * @param centroids Matrix in which centroids are stored.
   */
  void Cluster(const MatType& data,
               arma::Col<size_t>& assignments,
               arma::mat& centroids);
  
  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Set the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }
  
  //! Get the stop thresh.
  double StopThresh() const { return stopThresh; }
  //! Set the stop thresh.
  double& StopThresh() { return stopThresh; }
  
  //! Get the radius of the concerning points.
  double Radius() const { return radius; }
  //! Set the radius of the concerning points.
  double& Radius() { return radius; }
  
  //! Get the distance metric.
  const MetricType& Metric() const { return metric; }
  //! Modify the distance metric.
  MetricType& Metric() { return metric; }
  
 private:
  
  //! Maximum number of iterations before giving up.
  size_t maxIterations;
  
  /** If the 2-norm of the mean shift vector is less than stopThresh,
   *  iterations will terminate.
   */
  double stopThresh;
  
  /** When iterating, take points within distance of
   *     radius into consideration and two centroids within distance
   *     of radius will be ragarded as one centroid.
   */
  double radius;
  
  //! Instantiated distance metric.
  MetricType metric;
  
};

}; // namespace meanshift
}; // namespace mlpack

// Include implementation.
#include "mean_shift_impl.hpp"

#endif // __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP