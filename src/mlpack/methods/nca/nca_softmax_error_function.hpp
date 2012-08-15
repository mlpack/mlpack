/**
 * @file nca_softmax_error_function.hpp
 * @author Ryan Curtin
 *
 * Implementation of the stochastic neighbor assignment probability error
 * function (the "softmax error").
 *
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_HPP
#define __MLPACK_METHODS_NCA_NCA_SOFTMAX_ERROR_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nca {

/**
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
class SoftmaxErrorFunction
{
 public:
  /**
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

  /**
   * Evaluate the softmax function for the given covariance matrix.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   */
  double Evaluate(const arma::mat& covariance);

  /**
   * Evaluate the gradient of the softmax function for the given covariance
   * matrix.
   *
   * @param covariance Covariance matrix of Mahalanobis distance.
   * @param gradient Matrix to store the calculated gradient in.
   */
  void Gradient(const arma::mat& covariance, arma::mat& gradient);

  /**
   * Get the initial point.
   */
  const arma::mat GetInitialPoint() const;

 private:
  const arma::mat& dataset_;
  const arma::uvec& labels_;

  Kernel kernel_;

  arma::mat last_coordinates_;
  arma::mat stretched_dataset_;
  arma::vec p_; // Holds calculated p_i.
  arma::vec denominators_; // Holds denominators for calculation of p_ij.

  //! False is nothing has ever been precalculated (only at construction time).
  bool precalculated_;

  /**
   * Precalculate the denominators and numerators that will make up the p_ij,
   * but only if the coordinates matrix is different than the last coordinates
   * the Precalculate() method was run with.
   *
   * This will update last_coordinates_ and stretched_dataset_, and also
   * calculate the p_i and denominators_ which are used in the calculation of
   * p_i or p_ij.  The calculation will be O((n * (n + 1)) / 2), which is not
   * great.
   *
   * @param coordinates Coordinates matrix to use for precalculation.
   */
  void Precalculate(const arma::mat& coordinates);
};

}; // namespace nca
}; // namespace mlpack

// Include implementation.
#include "nca_softmax_error_function_impl.hpp"

#endif
