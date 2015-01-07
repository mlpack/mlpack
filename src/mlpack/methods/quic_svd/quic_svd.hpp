/**
 * @file quic_svd.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of QUIC-SVD.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_QUIC_SVD_QUIC_SVD_HPP
#define __MLPACK_METHODS_QUIC_SVD_QUIC_SVD_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cosine_tree/cosine_tree.hpp>

namespace mlpack {
namespace svd {

class QUIC_SVD
{
 public:

  /**
   * Constructor which implements the QUIC-SVD algorithm. The function calls the
   * CosineTree constructor to create a subspace basis, where the original
   * matrix's projection has minimum reconstruction error. The constructor then
   * uses the ExtractSVD() function to calculate the SVD of the original dataset
   * in that subspace.
   *
   * @param dataset Matrix for which SVD is calculated.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param sigma Diagonal matrix of singular values.
   * @param epsilon Error tolerance fraction for calculated subspace.
   * @param delta Cumulative probability for Monte Carlo error lower bound.
   */
  QUIC_SVD(const arma::mat& dataset,
           arma::mat& u,
           arma::mat& v,
           arma::mat& sigma,
           const double epsilon = 0.03,
           const double delta = 0.1);

  /**
   * This function uses the vector subspace created using a cosine tree to
   * calculate an approximate SVD of the original matrix.
   *
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param sigma Diagonal matrix of singular values.
   */
  void ExtractSVD(arma::mat& u,
                  arma::mat& v,
                  arma::mat& sigma);

 private:
  //! Matrix for which cosine tree is constructed.
  const arma::mat& dataset;
  //! Error tolerance fraction for calculated subspace.
  double epsilon;
  //! Cumulative probability for Monte Carlo error lower bound.
  double delta;
  //! Subspace basis of the input dataset.
  arma::mat basis;
};

}; // namespace svd
}; // namespace mlpack

// Include implementation.
#include "quic_svd_impl.hpp"

#endif
