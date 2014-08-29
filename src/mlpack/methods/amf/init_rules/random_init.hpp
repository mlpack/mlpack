/**
 * @file random_init.hpp
 * @author Mohan Rajendran
 *
 * Intialization rule for Non-Negative Matrix Factorization (NMF). This simple
 * initialization is performed by assigning a random matrix to W and H.
 *
 * This file is part of MLPACK 1.0.10.
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
#ifndef __MLPACK_METHODS_LMF_RANDOM_INIT_HPP
#define __MLPACK_METHODS_LMF_RANDOM_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

class RandomInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  RandomInitialization() { }

  template<typename MatType>
  inline static void Initialize(const MatType& V,
                                const size_t r,
                                arma::mat& W,
                                arma::mat& H)
  {
    // Simple implementation (left in the header file due to its simplicity).
    size_t n = V.n_rows;
    size_t m = V.n_cols;

    // Intialize to random values.
    W.randu(n, r);
    H.randu(r, m);
  }
};

}; // namespace amf
}; // namespace mlpack

#endif
