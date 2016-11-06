/**
 * @file mean_shift.hpp
 * @author Shangtong Zhang
 *
 * Mean Shift clustering
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP
#define MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <boost/utility.hpp>

namespace mlpack {
namespace meanshift /** Mean shift clustering. */ {

/**
 * This class implements mean shift clustering.  For each point in dataset,
 * apply mean shift algorithm until maximum iterations or convergence.  Then
 * remove duplicate centroids.
 *
 * A simple example of how to run mean shift clustering is shown below.
 *
 * @code
 * extern arma::mat data; // Dataset we want to run mean shift on.
 * arma::Col<size_t> assignments; // Cluster assignments.
 * arma::mat centroids; // Cluster centroids.
 *
 * MeanShift<> meanShift();
 * meanShift.Cluster(dataset, assignments, centroids);
 * @endcode
 *
 * @tparam UseKernel Use kernel or mean to calculate new centroid.
 *         If false, KernelType will be ignored.
 * @tparam KernelType The kernel to use.
 * @tparam MatType The type of matrix the data is stored in.
 */
template<bool UseKernel = false,
         typename KernelType = kernel::GaussianKernel,
         typename MatType = arma::mat>
class MeanShift
{
 public:
  /**
   * Create a mean shift object and set the parameters which mean shift will be
   * run with.
   *
   * @param radius If distance of two centroids is less than it, one will be
   *      removed. If this value isn't positive, an estimation will be given
   *      when clustering.
   * @param maxIterations Maximum number of iterations allowed before giving up
   *      iterations will terminate.
   * @param kernel Optional KernelType object.
   */
  MeanShift(const double radius = 0,
            const size_t maxIterations = 1000,
            const KernelType kernel = KernelType());

  /**
   * Give an estimation of radius based on given dataset.
   *
   * @param data Dataset for estimation.
   * @param ratio Percentage of dataset to use for nearest neighbor search.
   */
  double EstimateRadius(const MatType& data, const double ratio = 0.2);

  /**
   * Perform mean shift clustering on the data, returning a list of cluster
   * assignments and centroids.
   *
   * @tparam MatType Type of matrix.
   * @param data Dataset to cluster.
   * @param assignments Vector to store cluster assignments in.
   * @param centroids Matrix in which centroids are stored.
   */
  void Cluster(const MatType& data,
               arma::Col<size_t>& assignments,
               arma::mat& centroids,
               bool useSeeds = true);

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
   * To speed up, we can generate some seeds from data set and use
   * them as initial centroids rather than all the points in the data set.  The
   * basic idea here is that we will place our points into hypercube bins of
   * side length binSize, and any bins that contain fewer than minFreq points
   * will be removed as possible seeds.  Usually, 1 is a sufficient parameter
   * for minFreq, and the bin size can be set equal to the estimated radius.
   *
   * @param data The reference data set.
   * @param binSize Width of hypercube bins.
   * @param minFreq Minimum number of points in bin.
   * @param seed Matrix to store generated seeds in.
   */
  void GenSeeds(const MatType& data,
                const double binSize,
                const int minFreq,
                MatType& seeds);

  /**
   * Use kernel to calculate new centroid given dataset and valid neighbors.
   *
   * @param data The whole dataset
   * @param neighbors Valid neighbors
   * @param distances Distances to neighbors
   # @param centroid Store calculated centroid
   */
  template<bool ApplyKernel = UseKernel>
  typename std::enable_if<ApplyKernel, bool>::type
  CalculateCentroid(const MatType& data,
                    const std::vector<size_t>& neighbors,
                    const std::vector<double>& distances,
                    arma::colvec& centroid);

  /**
   * Use mean to calculate new centroid given dataset and valid neighbors.
   *
   * @param data The whole dataset
   * @param neighbors Valid neighbors
   * @param distances Distances to neighbors
   # @param centroid Store calculated centroid
   */
  template<bool ApplyKernel = UseKernel>
  typename std::enable_if<!ApplyKernel, bool>::type
  CalculateCentroid(const MatType& data,
                    const std::vector<size_t>& neighbors,
                    const std::vector<double>&, /*unused*/
                    arma::colvec& centroid);

  /**
   * If distance of two centroids is less than radius, one will be removed.
   * Points with distance to current centroid less than radius will be used
   * to calculate new centroid.
   */
  double radius;

  //! Maximum number of iterations before giving up.
  size_t maxIterations;

  //! Instantiated kernel.
  KernelType kernel;
};

} // namespace meanshift
} // namespace mlpack

// Include implementation.
#include "mean_shift_impl.hpp"

#endif // MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP
