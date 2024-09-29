/**
 * @file methods/amf/init_rules/random_init.hpp
 * @author Mohan Rajendran
 *
 * Initialization rule for alternating matrix factorization (AMF). This simple
 * initialization is performed by assigning a random matrix to W and H.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMF_RANDOM_INIT_HPP
#define MLPACK_METHODS_LMF_RANDOM_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This initialization rule for AMF simply fills the W and H matrices with
 * uniform random noise in [0, 1].
 */
class RandomAMFInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  RandomAMFInitialization() { }

  /**
   * Fill W and H with random uniform noise.
   *
   * @param V Input matrix.
   * @param r Rank of decomposition.
   * @param W W matrix, to be filled with random noise.
   * @param H H matrix, to be filled with random noise.
   */
  template<typename MatType, typename WHMatType>
  inline static void Initialize(const MatType& V,
                                const size_t r,
                                WHMatType& W,
                                WHMatType& H)
  {
    // Simple implementation (left in the header file due to its simplicity).
    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    // Initialize to random values.
    W.randu(n, r);
    H.randu(r, m);
  }

  /**
   * Fill W or H with random uniform noise.
   *
   * @param V Input matrix.
   * @param r Rank of decomposition.
   * @param M W or H matrix, to be filled with random noise.
   * @param whichMatrix If true, initialize W. Otherwise, initialize H.
   */
  template<typename MatType, typename WHMatType>
  inline void InitializeOne(const MatType& V,
                            const size_t r,
                            WHMatType& M,
                            const bool whichMatrix = true)
  {
    // Simple implementation (left in the header file due to its simplicity).
    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    // Initialize W or H to random values
    if (whichMatrix)
    {
      M.randu(n, r);
    }
    else
    {
      M.randu(r, m);
    }
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
