/**
 * @file mean_shift.hpp
 * @author Shangtong Zhang
 *
 * Mean Shift clustering
 */

#ifndef __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP
#define __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <boost/utility.hpp>

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
 *
 * MeanShift<> meanShift();
 * meanShift.Cluster(dataset, assignments, centroids);
 * @endcode
 *
 * @tparam KernelType the kernel to use.
 */
  
template<typename KernelType = kernel::GaussianKernel,
         typename MatType = arma::mat>
class MeanShift
{
 public:
  /**
   * Create a Mean Shift object and set the parameters which Mean Shift
   * will be run with.
   *
   * @param radius If distance of two centroids is less than it, one will be removed. If this value isn't positive, an estimation will be given when clustering.
   * @param maxIterations Maximum number of iterations allowed before giving up
   *        iterations will terminate.
   * @param kernel Optional KernelType object.
   */
  MeanShift(const double radius = 0,
            const size_t maxIterations = 1000,
            const KernelType kernel = KernelType());
  
  
  /**
   * Give an estimation of radius based on given dataset.
   * @param data Dataset for estimation.
   */
  double EstimateRadius(const MatType& data);
  
  /**
   * Perform Mean Shift clustering on the data, returning a list of cluster
   * assignments and centroids.
   * 
   * @tparam MatType Type of matrix.
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
  
  //! Get the radius.
  double Radius() const { return radius; }
  //! Set the radius.
  void Radius(double radius);
  
  //! Get the kernel.
  const KernelType& Kernel() const { return kernel; }
  //! Modify the kernel.
  KernelType& Kernel() { return kernel; }
  
 private:
  
  /**
   * If the kernel doesn't include a squared distance,
   * general way will be applied to calculate the weight of a data point.
   *
   * @param centroid The centroid to calculate the weight
   * @param point Calculate its weight
   * @param weight Store the weight
   * @return If true, the @point is near enough to the @centroid and @weight is valid,
   *         If false, the @point is far from the @centroid and @weight is invalid.
   */
  template <typename Kernel = KernelType>
  typename std::enable_if<!kernel::KernelTraits<Kernel>::UsesSquaredDistance, bool>::type
  CalcWeight(const arma::colvec& centroid, const arma::colvec& point, double& weight);
  
  /**
   * If the kernel includes a squared distance,
   * the weight of a data point can be calculated faster.
   *
   * @param centroid The centroid to calculate the weight
   * @param point Calculate its weight
   * @param weight Store the weight
   * @return If true, the @point is near enough to the @centroid and @weight is valid,
   *         If false, the @point is far from the @centroid and @weight is invalid.
   */
  template <typename Kernel = KernelType>
  typename std::enable_if<kernel::KernelTraits<Kernel>::UsesSquaredDistance, bool>::type
  CalcWeight(const arma::colvec& centroid, const arma::colvec& point, double& weight);
  
  /**
   * If distance of two centroids is less than radius, one will be removed.
   * Points with distance to current centroid less than radius will be used
   * to calculate new centroid.
   */
  double radius;
  
  // By storing radius * radius, we can speed up a little.
  double squaredRadius;
  
  //! Maximum number of iterations before giving up.
  size_t maxIterations;
  
  //! Instantiated kernel.
  KernelType kernel;
  
  metric::EuclideanDistance metric;
  
  
};

}; // namespace meanshift
}; // namespace mlpack

// Include implementation.
#include "mean_shift_impl.hpp"

#endif // __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP