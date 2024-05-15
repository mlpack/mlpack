/**
 * @file methods/amf/update_rules/nmf_mult_dist.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIST_UPDATE_RULES_HPP
#define MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIST_UPDATE_RULES_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The multiplicative distance update rules for matrices W and H. This follows
 * a method described in the following paper:
 *
 * @code
 * @inproceedings{lee2001algorithms,
 *   title={Algorithms for non-negative matrix factorization},
 *   author={Lee, D.D. and Seung, H.S.},
 *   booktitle={Advances in Neural Information Processing Systems 13
 *       (NIPS 2000)},
 *   pages={556--562},
 *   year={2001}
 * }
 * @endcode
 *
 * This is a multiplicative rule that ensures that the Frobenius norm
 * \f$ \sqrt{\sum_i \sum_j(V-WH)^2} \f$ is non-increasing between subsequent
 * iterations. Both of the update rules for W and H are defined in this file.
 */
class NMFMultiplicativeDistanceUpdate
{
 public:
  // Empty constructor required for the UpdateRule template.
  NMFMultiplicativeDistanceUpdate() { }

  /**
   * Initialize the factorization.  These update rules hold no information, so
   * the input parameters are ignored.
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
   * W_{ia} \leftarrow W_{ia} \frac{(VH^T)_{ia}}{(WHH^T)_{ia}}
   * \f]
   *
   * The function takes in all the matrices and only changes the value of the W
   * matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  template<typename MatType, typename WHMatType>
  inline static void WUpdate(const MatType& V,
                             WHMatType& W,
                             const WHMatType& H)
  {
    WHMatType zz = W * H * H.t();
    W = (W % (V * H.t())) / (W * H * H.t() + 1e-15);
  }

  /**
   * The update rule for the encoding matrix H. The formula used is
   *
   * \f[
   * H_{a\mu} \leftarrow H_{a\mu} \frac{(W^T V)_{a\mu}}{(W^T WH)_{a\mu}}
   * \f]
   *
   * The function takes in all the matrices and only changes the value of the H
   * matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix.
   * @param H Encoding matrix to be updated.
   */
  template<typename MatType, typename WHMatType>
  inline static void HUpdate(const MatType& V,
                             const WHMatType& W,
                             WHMatType& H)
  {
    WHMatType zz = W.t() * W * H;
    H = (H % (W.t() * V)) / (W.t() * W * H + 1e-15);
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
