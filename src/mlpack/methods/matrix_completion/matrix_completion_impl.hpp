/**
 * @file methods/matrix_completion/matrix_completion_impl.hpp
 * @author Stephen Tu
 *
 * Implementation of MatrixCompletion class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_IMPL_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_IMPL_HPP

#include "matrix_completion.hpp"
#include <mlpack/core/util/size_checks.hpp>

namespace mlpack {

inline MatrixCompletion::MatrixCompletion(
    const size_t m,
    const size_t n,
    const arma::umat& indices,
    const arma::vec& values,
    const size_t r) :
    m(m),
    n(n),
    indices(indices),
    values(values),
    sdp(indices.n_cols, 0, randu<arma::mat>(m + n, r))
{
  CheckValues();
  InitSDP();
}

inline MatrixCompletion::MatrixCompletion(
    const size_t m,
    const size_t n,
    const arma::umat& indices,
    const arma::vec& values,
    const arma::mat& initialPoint) :
    m(m),
    n(n),
    indices(indices),
    values(values),
    sdp(indices.n_cols, 0, initialPoint)
{
  CheckValues();
  InitSDP();
}

inline MatrixCompletion::MatrixCompletion(
    const size_t m,
    const size_t n,
    const arma::umat& indices,
    const arma::vec& values) :
    m(m),
    n(n),
    indices(indices),
    values(values),
    sdp(indices.n_cols, 0,
        randu<arma::mat>(m + n, DefaultRank(m, n, indices.n_cols)))
{
  CheckValues();
  InitSDP();
}

inline void MatrixCompletion::CheckValues()
{
  if (indices.n_rows != 2)
  {
    Log::Fatal << "MatrixCompletion::CheckValues(): matrix of constraint "
        << "indices does not have 2 rows!" << std::endl;
  }

  util::CheckSameSizes(indices, values, "MatrixCompletion::CheckValues()",
      "labels", false, true);

  for (size_t i = 0; i < values.n_elem; ++i)
  {
    if (indices(0, i) >= m || indices(1, i) >= n)
      Log::Fatal << "MatrixCompletion::CheckValues(): indices ("
          << indices(0, i) << ", " << indices(1, i)
          << ") are out of bounds for matrix of size " << m << " x n!"
          << std::endl;
  }
}

inline void MatrixCompletion::InitSDP()
{
  sdp.SDP().C().eye(m + n, m + n);
  sdp.SDP().SparseB() = 2. * values;
  const size_t p = indices.n_cols;
  for (size_t i = 0; i < p; ++i)
  {
    sdp.SDP().SparseA()[i].zeros(m + n, m + n);
    sdp.SDP().SparseA()[i](indices(0, i), m + indices(1, i)) = 1.;
    sdp.SDP().SparseA()[i](m + indices(1, i), indices(0, i)) = 1.;
  }
}

inline void MatrixCompletion::Recover(arma::mat& recovered)
{
  recovered = sdp.Function().GetInitialPoint();
  sdp.Optimize(recovered);
  recovered = recovered * trans(recovered);
  recovered = recovered(arma::span(0, m - 1), arma::span(m, m + n - 1));
}

inline size_t MatrixCompletion::DefaultRank(const size_t m,
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
  float r = 0.5 + std::sqrt(0.25 + 2 * p);
  if (ceil(r) > mpn)
    r = mpn; // An upper bound on the dimension.
  return ceil(r);
}

} // namespace mlpack

#endif
