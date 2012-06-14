/**
 * @file randominit.hpp
 * @author Mohan Rajendran
 *
 * Intialization rule for the Non-negative Matrix Factorization. This simple
 * initialization is performed by assigning a random matrix to W and H
 *
 */

#ifndef __MLPACK_METHODS_NMF_RANDOMINIT_HPP
#define __MLPACK_METHODS_NMF_RANDOMINIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nmf {

class RandomInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  RandomInitialization() { }

  inline static void Init(const arma::mat& V,
                     arma::mat& W,
                     arma::mat& H,
                     const size_t& r)
  {
    // Simple inplementation. This can be left here.
    size_t n = V.n_rows;
    size_t m = V.n_cols;
  
    // Intialize to random values
    W.randu(n,r);
    H.randu(r,m);
  }
  
}; // Class RandomInitialization

}; // namespace nmf
}; // namespace mlpack

#endif
