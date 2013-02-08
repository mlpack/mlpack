/**
 * @file als_update_rules.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization. This follows a method
 * titled 'Alternating Least Squares' described in the paper 'Positive Matrix
 * Factorization: A Non-negative Factor Model with Optimal Utilization of
 * Error Estimates of Data Values' by P. Paatero and U. Tapper. It uses least
 * squares projection formula to reduce the error value of
 * \f$ \sqrt{\sum_i \sum_j(V-WH)^2} \f$ by alternately calculating W and H
 * respectively while holding the other matrix constant.
 *
 * This file is part of MLPACK 1.0.4.
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
#ifndef __MLPACK_METHODS_NMF_ALS_UPDATE_RULES_HPP
#define __MLPACK_METHODS_NMF_ALS_UPDATE_RULES_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nmf {

/**
 * The update rule for the basis matrix W. The formula used is
 * \f[
 * W^T = \frac{HV^T}{HH^T}
 * \f]
 */
class WAlternatingLeastSquaresRule
{
 public:
  // Empty constructor required for the WUpdateRule template.
  WAlternatingLeastSquaresRule() { }

  /**
   * The update function that actually updates the W matrix. The function takes
   * in all the matrices and only changes the value of the W matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  inline static void Update(const arma::mat& V,
                            arma::mat& W,
                            const arma::mat& H)
  {
    // The call to inv() sometimes fails; so we are using the psuedoinverse.
    // W = (inv(H * H.t()) * H * V.t()).t();
    W = V * H.t() * pinv(H * H.t());

    // Set all negative numbers to machine epsilon
    for (size_t i = 0; i < W.n_elem; i++)
    {
      if (W(i) < 0.0)
      {
        W(i) = 0.0;
      }
    }
  }
};

/**
 * The update rule for the encoding matrix H. The formula used is
 * \f[
 * H = \frac{W^TV}{W^TW}
 * \f]
 */
class HAlternatingLeastSquaresRule
{
 public:
  // Empty constructor required for the HUpdateRule template.
  HAlternatingLeastSquaresRule() { }

  /**
   * The update function that actually updates the H matrix. The function takes
   * in all the matrices and only changes the value of the H matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix.
   * @param H Encoding matrix to be updated.
   */
  inline static void Update(const arma::mat& V,
                            const arma::mat& W,
                            arma::mat& H)
  {
    H = pinv(W.t() * W) * W.t() * V;

    // Set all negative numbers to 0.
    for (size_t i = 0; i < H.n_elem; i++)
    {
      if (H(i) < 0.0)
      {
        H(i) = 0.0;
      }
    }
  }
};

}; // namespace nmf
}; // namespace mlpack

#endif
