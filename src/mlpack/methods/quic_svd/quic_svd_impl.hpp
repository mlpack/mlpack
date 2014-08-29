/**
 * @file quic_svd_impl.hpp
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
#ifndef __MLPACK_METHODS_QUIC_SVD_QUIC_SVD_IMPL_HPP
#define __MLPACK_METHODS_QUIC_SVD_QUIC_SVD_IMPL_HPP

// In case it hasn't been included yet.
#include "quic_svd.hpp"

using namespace mlpack::tree;

namespace mlpack {
namespace svd {

QUIC_SVD::QUIC_SVD(const arma::mat& dataset,
                   arma::mat& u,
                   arma::mat& v,
                   arma::mat& sigma,
                   const double epsilon,
                   const double delta) :
    dataset(dataset),
    epsilon(epsilon),
    delta(delta)
{
  // Since columns are sample in the implementation, the matrix is transposed if
  // necessary for maximum speedup.
  CosineTree* ctree;
  if (dataset.n_cols > dataset.n_rows)
    ctree = new CosineTree(dataset, epsilon, delta);
  else
    ctree = new CosineTree(dataset.t(), epsilon, delta);

  // Get subspace basis by creating the cosine tree.
  ctree->GetFinalBasis(basis);

  // Use the ExtractSVD algorithm mentioned in the paper to extract the SVD of
  // the original dataset in the obtained subspace.
  ExtractSVD(u, v, sigma);
}

void QUIC_SVD::ExtractSVD(arma::mat& u,
                          arma::mat& v,
                          arma::mat& sigma)
{
  // Calculate A * V_hat, necessary for further calculations.
  arma::mat projectedMat;
  if (dataset.n_cols > dataset.n_rows)
    projectedMat = dataset.t() * basis;
  else
    projectedMat = dataset * basis;

  // Calculate the squared projected matrix.
  arma::mat projectedMatSquared = projectedMat.t() * projectedMat;

  // Calculate the SVD of the above matrix.
  arma::mat uBar, vBar;
  arma::vec sigmaBar;
  arma::svd(uBar, sigmaBar, vBar, projectedMatSquared);

  // Calculate the approximate SVD of the original matrix, using the SVD of the
  // squared projected matrix.
  v = basis * vBar;
  sigma = arma::sqrt(diagmat(sigmaBar));
  u = projectedMat * vBar * sigma.i();

  // Since columns are sampled, the unitary matrices have to be exchanged, if
  // the transposed matrix is not passed.
  if (dataset.n_cols > dataset.n_rows)
  {
    arma::mat tempMat = u;
    u = v;
    v = tempMat;
  }
}

}; // namespace svd
}; // namespace mlpack

#endif
