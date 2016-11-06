/**
 * @file nmf_als.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMF_UPDATE_RULES_NMF_ALS_HPP
#define MLPACK_METHODS_LMF_UPDATE_RULES_NMF_ALS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This class implements a method titled 'Alternating Least Squares' described
 * in the following paper:
 *
 * @code
 * @article{paatero1994positive,
 *  title={Positive matrix factorization: A non-negative factor model with
 *      optimal utilization of error estimates of data values},
 *  author={Paatero, P. and Tapper, U.},
 *  journal={Environmetrics},
 *  volume={5},
 *  number={2},
 *  pages={111--126},
 *  year={1994}
 * }
 * @endcode
 *
 * It uses the least squares projection formula to reduce the error value of
 * \f$ \sqrt{\sum_i \sum_j(V-WH)^2} \f$ by alternately calculating W and H
 * respectively while holding the other matrix constant.
 */
class NMFALSUpdate
{
 public:
  //! Empty constructor required for the UpdateRule template.
  NMFALSUpdate() { }

  /**
   * Set initial values for the factorization.  In this case, we don't need to
   * set anything.
   */
  template<typename MatType>
  void Initialize(const MatType& /* dataset */, const size_t /* rank */)
  {
    // Nothing to do.
  }

  /**
   * The update rule for the basis matrix W. The formula used isa
   *
   * \f[
   * W^T = \frac{H V^T}{H H^T}
   * \f]
   *
   * The function takes in all the matrices and only changes the value of the W
   * matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  template<typename MatType>
  inline static void WUpdate(const MatType& V,
                             arma::mat& W,
                             const arma::mat& H)
  {
    // The call to inv() sometimes fails; so we are using the psuedoinverse.
    // W = (inv(H * H.t()) * H * V.t()).t();
    W = V * H.t() * pinv(H * H.t());

    // Set all negative numbers to machine epsilon.
    for (size_t i = 0; i < W.n_elem; i++)
    {
      if (W(i) < 0.0)
      {
        W(i) = 0.0;
      }
    }
  }

  /**
   * The update rule for the encoding matrix H. The formula used is
   *
   * \f[
   * H = \frac{W^T V}{W^T W}
   * \f]
   *
   * The function takes in all the matrices and only changes the value of the H
   * matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix.
   * @param H Encoding matrix to be updated.
   */
  template<typename MatType>
  inline static void HUpdate(const MatType& V,
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

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
}; // class NMFALSUpdate

} // namespace amf
} // namespace mlpack

#endif
