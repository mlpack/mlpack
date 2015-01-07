/**
 * @file random_init.hpp
 * @author Mohan Rajendran
 *
 * Intialization rule for Non-Negative Matrix Factorization (NMF). This simple
 * initialization is performed by assigning a random matrix to W and H.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
