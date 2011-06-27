/***
 * @file nca_softmax.h
 * @author Ryan Curtin
 *
 * Implementation of the stochastic neighbor assignment probability error
 * function (the "softmax error").
 */
#ifndef __MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_H
#define __MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_H

#include <armadillo>
#include <map>

namespace mlpack {
namespace nca {

/***
 * The "softmax" stochastic neighbor assignment probability function.
 *
 * The actual function is
 *
 * p_ij = (exp(-|| A x_i - A x_j || ^ 2)) /
 *     (sum_{k != i} (exp(-|| A x_i - A x_k || ^ 2)))
 *
 * where x_n represents a point and A is the current scaling matrix.
 *
 * This class is more flexible than the original paper, allowing an arbitrary
 * kernel function to be used, meaning that the Mahalanobis distance is not the
 * only allowed way to run NCA.  However, the Mahalanobis distance is probably
 * the best way to use this.
 */
template<typename Kernel>
class SoftmaxErrorFunction {
 public:
  /***
   * Initialize with the given kernel; useful when the kernel has some state to
   * store, which is set elsewhere.  If no kernel is given, an empty kernel is
   * used; this way, you can call the constructor with no arguments.  A
   * reference to the dataset we will be optimizing over is also required.
   * 
   * @param dataset Matrix containing the dataset.
   * @param labels Vector of class labels for each point in the dataset.
   * @param kernel Instantiated kernel (optional).
   */
  SoftmaxErrorFunction(const arma::mat& dataset,
                       const arma::uvec& labels,
                       Kernel kernel = Kernel());

  /***
   * Evaluate the softmax function for the given covariance matrix.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   */
  double Evaluate(const arma::mat& covariance);

  /***
   * Evaluate the gradient of the softmax function for the given covariance
   * matrix.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param gradient Matrix to store the calculated gradient in.
   */
  void Gradient(const arma::mat& covariance, arma::mat& gradient);

  /***
   * Get the initial point.
   */
  arma::mat GetInitialPoint();

 private:
  const arma::mat& dataset_;
  const arma::uvec& labels_;

  Kernel kernel_;

  std::multimap<index_t, index_t> class_mapping_;
};

}; // namespace nca
}; // namespace mlpack

// Include implementation.
#include "nca_softmax_error_function_impl.h"

#endif
