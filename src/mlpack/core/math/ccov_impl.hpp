/**
 * @file ccov_impl.hpp
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ColumnCovariance(X) is same as cov(trans(X)) but without the cost of computing trans(X)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_CCOV_IMPL_HPP
#define MLPACK_CORE_MATH_CCOV_IMPL_HPP

#include "ccov.hpp"

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

template<typename eT>
inline
arma::Mat<eT>
ColumnCovariance(const arma::Mat<eT>& A, const size_t norm_type)
{
  if (norm_type > 1)
  {
    Log::Fatal << "ColumnCovariance(): norm_type must be 0 or 1" << std::endl;
  }

  arma::Mat<eT> out;

  if (A.is_vec())
  {
    if (A.n_rows == 1)
    {
      out = arma::var(arma::trans(A), norm_type);
    }
    else
    {
      out = arma::var(A, norm_type);
    }
  }
  else
  {
    const size_t N = A.n_cols;
    const eT norm_val = (norm_type == 0) ?
        ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    const arma::Col<eT> acc = arma::sum(A, 1);

    out = A * arma::trans(A);
    out -= (acc * arma::trans(acc)) / eT(N);
    out /= norm_val;
  }

  return out;
}

template<typename T>
inline
arma::Mat< std::complex<T> >
ColumnCovariance(const arma::Mat< std::complex<T> >& A,
     const size_t norm_type)
{
  if (norm_type > 1)
  {
    Log::Fatal << "ColumnCovariance(): norm_type must be 0 or 1" << std::endl;
  }

  typedef typename std::complex<T> eT;

  arma::Mat<eT> out;

  if (A.is_vec())
  {
    if (A.n_rows == 1)
    {
      const arma::Mat<T> tmp_mat = arma::var(arma::trans(A), norm_type);
      out.set_size(1, 1);
      out[0] = tmp_mat[0];
    }
    else
    {
      const arma::Mat<T> tmp_mat = arma::var(A, norm_type);
      out.set_size(1, 1);
      out[0] = tmp_mat[0];
    }
  }
  else
  {
    const size_t N = A.n_cols;
    const eT norm_val = (norm_type == 0) ?
        ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    const arma::Col<eT> acc = arma::sum(A, 1);

    out = A * arma::trans(arma::conj(A));
    out -= (acc * arma::trans(arma::conj(acc))) / eT(N);
    out /= norm_val;
  }

  return out;
}

} // namespace math
} // namespace mlpack


#endif // MLPACK_CORE_MATH_CCOV_IMPL_HPP
