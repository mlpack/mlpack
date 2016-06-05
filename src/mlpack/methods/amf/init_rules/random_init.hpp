/**
 * @file random_init.hpp
 * @author Mohan Rajendran
 *
 * Initialization rule for alternating matrix factorization (AMF). This simple
 * initialization is performed by assigning a random matrix to W and H.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_METHODS_LMF_RANDOM_INIT_HPP
#define MLPACK_METHODS_LMF_RANDOM_INIT_HPP

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
