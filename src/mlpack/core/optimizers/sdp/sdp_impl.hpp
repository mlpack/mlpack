/**
 * @file sdp_impl.hpp
 * @author Stephen Tu
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_SDP_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_SDP_IMPL_HPP

#include "sdp.hpp"

namespace mlpack {
namespace optimization {

template <typename ObjectiveMatrixType>
SDP<ObjectiveMatrixType>::SDP() :
    c(),
    sparseA(),
    sparseInequalityA(),
    sparseB(),
    denseA(),
    denseInequalityA(),
    denseB()
{ /* Nothing to do. */ }

template <typename ObjectiveMatrixType>
SDP<ObjectiveMatrixType>::SDP(const size_t n,
                              const size_t numSparseConstraints,
                              const size_t numSparseInequalityConstraints,
                              const size_t numDenseConstraints,
                              const size_t numDenseInequalityConstraints) :
    c(n, n),
    sparseA(numSparseConstraints),
    sparseInequalityA(numSparseInequalityConstraints),
    sparseB(numSparseConstraints),
    denseA(numDenseConstraints),
    denseInequalityA(numDenseInequalityConstraints),
    denseB(numDenseConstraints)
{
  for (size_t i = 0; i < numSparseConstraints; i++)
    sparseA[i].zeros(n, n);
  for (size_t i = 0; i < numSparseConstraints; i++)
    sparseInequalityA[i].zeros(n, n);
  for (size_t i = 0; i < numDenseConstraints; i++)
    denseA[i].zeros(n, n);
  for (size_t i = 0; i < numDenseConstraints; i++)
    denseInequalityA[i].zeros(n, n);
}

template <typename ObjectiveMatrixType>
bool SDP<ObjectiveMatrixType>::HasLinearlyIndependentConstraints() const
{
  // Very inefficient, should only be used for testing/debugging

  const size_t n2bar = N2bar();
  arma::mat A(NumConstraints(), n2bar);
  if (A.n_rows > n2bar)
    return false;

  for (size_t i = 0; i < SparseA().n_elem; i++)
  {
    arma::vec sa;
    math::Svec(arma::mat(SparseA()[i]), sa);
    A.row(i) = sa.t();
  }

  for (size_t i = 0; i < SparseInequalityA().n_elem; i++)
  {
    arma::vec sa;
    math::Svec(arma::mat(SparseInequalityA()[i]), sa);
    A.row(SparseA().n_elem + i) = sa.t();
  }

  for (size_t i = 0; i < DenseA().n_elem; i++)
  {
    arma::vec sa;
    math::Svec(DenseA()[i], sa);
    A.row(NumSparseConstraints() + i) = sa.t();
  }

  for (size_t i = 0; i < DenseInequalityA().n_elem; i++)
  {
    arma::vec sa;
    math::Svec(DenseInequalityA()[i], sa);
    A.row(NumSparseConstraints() + DenseA().n_elem + i) = sa.t();
  }

  const arma::vec s = arma::svd(A);
  return s(s.n_elem - 1) > 1e-5;
}

} // namespace optimization
} // namespace mlpack

#endif
