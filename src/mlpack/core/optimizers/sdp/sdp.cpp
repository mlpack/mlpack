/**
 * @file sdp.cpp
 * @author Stephen Tu
 *
 */

#include "sdp.hpp"

namespace mlpack {
namespace optimization {

SDP::SDP(const size_t n,
         const size_t numSparseConstraints,
         const size_t numDenseConstraints) :
    n(n),
    sparseC(n, n),
    denseC(n, n),
    hasModifiedSparseObjective(false),
    hasModifiedDenseObjective(false),
    sparseA(numSparseConstraints),
    sparseB(numSparseConstraints),
    denseA(numDenseConstraints),
    denseB(numDenseConstraints)
{
  denseC.zeros();
}

bool SDP::HasLinearlyIndependentConstraints() const
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
