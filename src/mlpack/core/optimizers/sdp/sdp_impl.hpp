/**
 * @file sdp_impl.hpp
 * @author Stephen Tu
 *
 *
 * This file is part of mlpack 2.0.2.
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
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_SDP_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_SDP_IMPL_HPP

#include "sdp.hpp"

namespace mlpack {
namespace optimization {

template <typename ObjectiveMatrixType>
SDP<ObjectiveMatrixType>::SDP() :
    c(),
    sparseA(),
    sparseB(),
    denseA(),
    denseB()
{

}

template <typename ObjectiveMatrixType>
SDP<ObjectiveMatrixType>::SDP(const size_t n,
                              const size_t numSparseConstraints,
                              const size_t numDenseConstraints) :
    c(n, n),
    sparseA(numSparseConstraints),
    sparseB(numSparseConstraints),
    denseA(numDenseConstraints),
    denseB(numDenseConstraints)
{
  for (size_t i = 0; i < numSparseConstraints; i++)
    sparseA[i].zeros(n, n);
  for (size_t i = 0; i < numDenseConstraints; i++)
    denseA[i].zeros(n, n);
}

template <typename ObjectiveMatrixType>
bool SDP<ObjectiveMatrixType>::HasLinearlyIndependentConstraints() const
{
  // Very inefficient, should only be used for testing/debugging

  const size_t n2bar = N2bar();
  arma::mat A(NumConstraints(), n2bar);
  if (A.n_rows > n2bar)
    return false;

  for (size_t i = 0; i < NumSparseConstraints(); i++)
  {
    arma::vec sa;
    math::Svec(arma::mat(SparseA()[i]), sa);
    A.row(i) = sa.t();
  }
  for (size_t i = 0; i < NumDenseConstraints(); i++)
  {
    arma::vec sa;
    math::Svec(DenseA()[i], sa);
    A.row(NumSparseConstraints() + i) = sa.t();
  }

  const arma::vec s = arma::svd(A);
  return s(s.n_elem - 1) > 1e-5;
}

} // namespace optimization
} // namespace mlpack

#endif
