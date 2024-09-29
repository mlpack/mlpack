/**
 * @file methods/amf/update_rules/nmf_mult_div.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIV_HPP
#define MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIV_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This follows a method described in the paper 'Algorithms for Non-negative
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
 * This is a multiplicative rule that ensures that the Kullback-Leibler
 * divergence
 *
 * \f[
 * \sum_i \sum_j (V_{ij} \log\frac{V_{ij}}{(W H)_{ij}} - V_{ij} + (W H)_{ij})
 * \f]
 *
 * is non-increasing between subsequent iterations. Both of the update rules
 * for W and H are defined in this file.
 *
 * This set of update rules is not meant to work with sparse matrices.  Using
 * sparse matrices often causes NaNs in the output, so other choices of update
 * rules are better in that situation.
 */
class NMFMultiplicativeDivergenceUpdate
{
 public:
  // Empty constructor required for the WUpdateRule template.
  NMFMultiplicativeDivergenceUpdate() { }

  /**
   * Initialize the factorization.  These rules don't store any state, so the
   * input values are ignore.
   */
  template<typename MatType>
  void Initialize(const MatType& /* dataset */, const size_t /* rank */)
  {
    // Nothing to do.
  }

  /**
   * The update rule for the basis matrix W. The formula used is
   *
   * \f[
   * W_{ia} \leftarrow W_{ia} \frac{\sum_{\mu} H_{a\mu} V_{i\mu} / (W H)_{i\mu}}
   * {\sum_{\nu} H_{a\nu}}
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
    // Simple implementation left in the header file.
    W %= ((V / (W * H + 1e-15)) * H.t()) /
        (repmat(sum(H, 1).t(), W.n_rows, 1) + 1e-15);
  }

  /**
   * The update rule for the encoding matrix H. The formula used is
   *
   * \f[
   * H_{a\mu} \leftarrow H_{a\mu} \frac{\sum_{i} W_{ia} V_{i\mu}/(WH)_{i\mu}}
   * {\sum_{k} H_{ka}}
   * \f]
   *
   * The function takes in all the matrices and only changes the value of the H
   * matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix.
   * @param H Encoding matrix to updated.
   */
  template<typename MatType, typename WHMatType>
  inline static void HUpdate(const MatType& V,
                             const WHMatType& W,
                             WHMatType& H)
  {
    // Simple implementation left in the header file.
    H %= (W.t() * (V / (W * H + 1e-15))) /
        (repmat(sum(W, 0).t(), 1, H.n_cols) + 1e-15);
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
