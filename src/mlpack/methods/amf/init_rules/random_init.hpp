/**
 * @file random_init.hpp
 * @author Mohan Rajendran
 *
 * Initialization rule for alternating matrix forization (AMF). This simple
 * initialization is performed by assigning a random matrix to W and H.
 */
#ifndef __MLPACK_METHODS_LMF_RANDOM_INIT_HPP
#define __MLPACK_METHODS_LMF_RANDOM_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This initialization rule for AMF simply fills the W and H matrices with
 * uniform random noise in [0, 1].
 */
class RandomInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  RandomInitialization() { }

  /**
   * Fill W and H with random uniform noise.
   *
   * @param V Input matrix.
   * @param r Rank of decomposition.
   * @param W W matrix, to be filled with random noise.
   * @param H H matrix, to be filled with random noise.
   */
  template<typename MatType>
  inline static void Initialize(const MatType& V,
                                const size_t r,
                                arma::mat& W,
                                arma::mat& H)
  {
    // Simple implementation (left in the header file due to its simplicity).
    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    // Initialize to random values.
    W.randu(n, r);
    H.randu(r, m);
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace amf
} // namespace mlpack

#endif
