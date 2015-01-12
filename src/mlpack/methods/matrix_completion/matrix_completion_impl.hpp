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
                                   const arma::umat& indices,
                                   const arma::vec& values,
                                   const size_t r)
  : m(m), n(n), indices(indices), values(values),
    sdp(indices.n_cols, 0, arma::randu<arma::mat>(m + n, r))
{
  checkValues();
  initSdp();
}

MatrixCompletion::MatrixCompletion(const size_t m,
                                   const size_t n,
                                   const arma::umat& indices,
                                   const arma::vec& values,
                                   const arma::mat& initialPoint)
  : m(m), n(n), indices(indices), values(values),
    sdp(indices.n_cols, 0, initialPoint)
{
  checkValues();
  initSdp();
}

MatrixCompletion::MatrixCompletion(const size_t m,
                                   const size_t n,
                                   const arma::umat& indices,
                                   const arma::vec& values)
  : m(m), n(n), indices(indices), values(values),
    sdp(indices.n_cols, 0,
        arma::randu<arma::mat>(m + n,
                               DefaultRank(m, n, indices.n_cols)))
{
  checkValues();
  initSdp();
}

void MatrixCompletion::checkValues()
{
  if (indices.n_rows != 2)
    Log::Fatal << "matrix of constraint indices does not have 2 rows"
      << std::endl;

  if (indices.n_cols != values.n_elem)
    Log::Fatal << "the number of constraint indices "
      << "(columns of constraint indices matrix) "
      << "does not match the number of constraint values "
      << "(length of constraint value vector)"
      << std::endl;

  for (size_t i = 0; i < values.n_elem; i++)
  {
    if (indices(0, i) >= m || indices(1, i) >= n)
      Log::Fatal << "indices (" << indices(0, i) << ", " << indices(1, i) << ")"
        << " are out of bounds for matrix of size " << m << " x " << "n"
        << std::endl;
  }
}

void MatrixCompletion::initSdp()
{
  sdp.SparseC().eye(m + n, m + n);
  sdp.SparseB() = 2. * values;
  const size_t p = indices.n_cols;
  for (size_t i = 0; i < p; i++)
  {
    sdp.SparseA()[i].zeros(m + n, m + n);
    sdp.SparseA()[i](indices(0, i), m + indices(1, i)) = 1.;
    sdp.SparseA()[i](m + indices(1, i), indices(0, i)) = 1.;
  }
}

void MatrixCompletion::Recover(arma::mat& recovered)
{
  recovered = sdp.Function().GetInitialPoint();
  sdp.Optimize(recovered);
  recovered = recovered * trans(recovered);
  recovered = recovered(arma::span(0, m - 1), arma::span(m, m + n - 1));
}

size_t MatrixCompletion::DefaultRank(const size_t m,
                                     const size_t n,
                                     const size_t p)
{
  // If r = O(sqrt(p)), then we are guaranteed an exact solution.
  // For more details, see
  //
  //   On the rank of extreme matrices in semidefinite programs and the
  //   multiplicity of optimal eigenvalues.
  //   Pablo Moscato, Michael Norman, and Gabor Pataki.
  //   Math Oper. Res., 23(2). 1998.

  const size_t mpn = m + n;
  float r = 0.5 + sqrt(0.25 + 2 * p);
  if (ceil(r) > mpn)
    r = mpn; // An upper bound on the dimension.
  return ceil(r);
}

} // namespace matrix_completion
} // namespace mlpack

#endif
