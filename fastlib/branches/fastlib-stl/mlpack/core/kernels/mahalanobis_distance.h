/***
 * @file mahalanobis_dstance.h
 * @author Ryan Curtin
 *
 * The Mahalanobis distance.
 */
#ifndef __MLPACK_CORE_KERNELS_MAHALANOBIS_DISTANCE_H
#define __MLPACK_CORE_KERNELS_MAHALANOBIS_DISTANCE_H

#include <armadillo>

namespace mlpack {
namespace kernel {

/***
 * The Mahalanobis distance, which is a stretched Euclidean distance:
 *
 * d(x_1, x_2) = (x_1 - x_2)^T Q (x_1 - x_2)
 *
 * where Q is the covariance matrix.
 *
 * Because each evaluation multiplies (x_1 - x_2) by the covariance matrix, it
 * may be much quicker to use an LMetric and simply stretch the actual dataset
 * itself before performing any evaluations.  However, this class is provided
 * for convenience.
 *
 * @tparam t_take_root If true, takes the root of the output.
 */
template<bool t_take_root = false>
class MahalanobisDistance {
 public:
  /***
   * Initialize the Mahalanobis distance with the identity matrix as covariance.
   * Because we don't actually know the size of the vectors we will be using, we
   * delay creation of the covariance matrix until evaluation.
   */
  MahalanobisDistance() : covariance_(0, 0) { }

  /***
   * Initialize the Mahalanobis distance with the given covariance matrix.
   */
  MahalanobisDistance(arma::mat& covariance) : covariance_(covariance) { }

  /***
   * Evaluate the distance between the two given points using this Mahalanobis
   * distance.
   */
  double Evaluate(arma::vec& a, arma::vec& b);

  /***
   * Return the covariance matrix.
   */
  const arma::mat& GetCovariance() const { return covariance_; }
  arma::mat& GetCovariance() { return covariance_; }

 private:
  arma::mat covariance_;
};

}; // namespace kernel
}; // namespace mlpack

#include "mahalanobis_distance_impl.h"

#endif
