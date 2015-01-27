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
 * MeanShift<arma::mat, kernel::GaussianKernel> meanShift();
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
   * @param duplicateThresh If distance of two centroids is less than it, one will be removed. If this value is negative, an estimation will be given when clustering.
   * @param maxIterations Maximum number of iterations allowed before giving up
   * @param stopThresh If the 2-norm of the mean shift vector is less than stopThresh, 
   *        iterations will terminate.
   * @param kernel Optional KernelType object.
   */
  MeanShift(const double duplicateThresh = -1,
            const size_t maxIterations = 1000,
            const double stopThresh = 1e-3,
            const KernelType kernel = KernelType());
  
  
  /**
   * Give an estimation of duplicate thresh based on given dataset.
   * @param data Dataset for estimation.
   */
  double estimateDuplicateThresh(const MatType& data);
  
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
  
  //! Get the kernel.
  const KernelType& Kernel() const { return kernel; }
  //! Modify the kernel.
  KernelType& Kernel() { return kernel; }
  
  //! Get the duplicate thresh.
  double DuplicateThresh() const { return duplicateThresh; }
  //! Set the duplicate thresh.
  double& DuplicateThresh() { return duplicateThresh; }
  
 private:
  
  // If distance of two centroids is less than duplicateThresh, one will be removed.
  double duplicateThresh;
  
  //! Maximum number of iterations before giving up.
  size_t maxIterations;
  
  /** 
   * If the 2-norm of the mean shift vector is less than stopThresh,
   *  iterations will terminate.
   */
  double stopThresh;
  
  //! Instantiated kernel.
  KernelType kernel;
  
};

}; // namespace meanshift
}; // namespace mlpack

// Include implementation.
#include "mean_shift_impl.hpp"

#endif // __MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP