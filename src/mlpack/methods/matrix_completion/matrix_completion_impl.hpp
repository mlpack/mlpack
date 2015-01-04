/**
 * @file matrix_completion_impl.hpp
 * @author Stephen Tu
 *
 * Implementation of MatrixCompletion class.
 */
#ifndef __MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_IMPL_HPP
#define __MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_IMPL_HPP

namespace mlpack {
namespace matrix_completion {

MatrixCompletion::MatrixCompletion(const size_t m,
                                   const size_t n,
                                   const arma::mat& entries,
                                   const size_t r)
  : m(m), n(n), entries(entries),
    sdp(entries.n_cols, 0, CreateInitialPoint(m, n, r))
{
  initSdp();
}

MatrixCompletion::MatrixCompletion(const size_t m,
                                   const size_t n,
                                   const arma::mat& entries,
                                   const arma::mat& initialPoint)
  : m(m), n(n), entries(entries),
    sdp(entries.n_cols, 0, initialPoint)
{
  initSdp();
}

MatrixCompletion::MatrixCompletion(const size_t m,
                                   const size_t n,
                                   const arma::mat& entries)
  : m(m), n(n), entries(entries),
    sdp(entries.n_cols, 0, CreateInitialPoint(m, n, DefaultRank(m, n, entries.n_cols)))
{
  initSdp();
}

void MatrixCompletion::initSdp()
{
  sdp.SparseC().eye(m + n, m + n);
  sdp.SparseB() = 2. * entries.row(2);
  const size_t p = entries.n_cols;
  for (size_t i = 0; i < p; i++)
  {
    sdp.SparseA()[i].zeros(m + n, m + n);
    sdp.SparseA()[i](entries(0, i), m + entries(1, i)) = 1.;
    sdp.SparseA()[i](m + entries(1, i), entries(0, i)) = 1.;
  }
}

void MatrixCompletion::Recover()
{
  recovered = sdp.Function().GetInitialPoint();
  sdp.Optimize(recovered);
}

size_t MatrixCompletion::DefaultRank(const size_t m,
                                     const size_t n,
                                     const size_t p)
{
  const size_t mpn = m + n;
  float r = 0.5 + sqrt(0.25 + 2 * p);
  if (ceil(r) > mpn)
    r = mpn; // An upper bound on the dimension.
  return ceil(r);
}

arma::mat MatrixCompletion::CreateInitialPoint(const size_t m,
                                               const size_t n,
                                               const size_t r)
{
  const size_t mpn = m + n;
  return arma::randu<arma::mat>(mpn, r);
}

} // namespace matrix_completion
} // namespace mlpack

#endif
