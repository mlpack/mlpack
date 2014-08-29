/**
 * @file quic_svd.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of QUIC-SVD.
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
