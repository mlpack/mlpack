/**
 * @file methods/quic_svd/quic_svd_impl.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of QUIC-SVD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_QUIC_SVD_QUIC_SVD_IMPL_HPP
#define MLPACK_METHODS_QUIC_SVD_QUIC_SVD_IMPL_HPP

// In case it hasn't been included yet.
#include "quic_svd.hpp"

namespace mlpack {

template<typename MatType>
inline QUIC_SVD<MatType>::QUIC_SVD(
    const MatType& dataset,
    MatType& u,
    MatType& v,
    MatType& sigma,
    const double epsilon,
    const double delta)
{
  Apply(dataset, u, v, sigma, epsilon, delta);
}

template<typename MatType>
inline QUIC_SVD<MatType>::QUIC_SVD(
    const double /* epsilon */,
    const double /* delta */)
{
  /* Nothing to do here */
}

template<typename MatType>
inline void QUIC_SVD<MatType>::Apply(
    const MatType& dataset,
    MatType& u,
    MatType& v,
    MatType& sigma,
    const double epsilon,
    const double delta)
{
  // Since columns are sample in the implementation, the matrix is transposed if
  // necessary for maximum speedup.
  CosineTree<MatType>* ctree;
  if (dataset.n_cols > dataset.n_rows)
    ctree = new CosineTree<MatType>(dataset, epsilon, delta);
  else
    ctree = new CosineTree<MatType>(dataset.t(), epsilon, delta);

  // Get subspace basis by creating the cosine tree.
  ctree->GetFinalBasis(basis);

  // Delete cosine tree.
  delete ctree;

  // Use the ExtractSVD algorithm mentioned in the paper to extract the SVD of
  // the original dataset in the obtained subspace.
  ExtractSVD(dataset, u, v, sigma);
}

template<typename MatType>
inline void QUIC_SVD<MatType>::ExtractSVD(const MatType& dataset,
                                          MatType& u,
                                          MatType& v,
                                          MatType& sigma)
{
  // Calculate A * V_hat, necessary for further calculations.
  MatType projectedMat;
  if (dataset.n_cols > dataset.n_rows)
    projectedMat = dataset.t() * basis;
  else
    projectedMat = dataset * basis;

  // Calculate the squared projected matrix.
  MatType projectedMatSquared = projectedMat.t() * projectedMat;

  // Calculate the SVD of the above matrix.
  MatType uBar, vBar;
  arma::Col<typename MatType::elem_type> sigmaBar;
  arma::svd(uBar, sigmaBar, vBar, projectedMatSquared);

  // Calculate the approximate SVD of the original matrix, using the SVD of the
  // squared projected matrix.
  v = basis * vBar;
  sigma = sqrt(diagmat(sigmaBar));
  u = projectedMat * vBar * sigma.i();

  // Since columns are sampled, the unitary matrices have to be exchanged, if
  // the transposed matrix is not passed.
  if (dataset.n_cols > dataset.n_rows)
  {
    MatType tempMat = u;
    u = v;
    v = tempMat;
  }
}

} // namespace mlpack

#endif
