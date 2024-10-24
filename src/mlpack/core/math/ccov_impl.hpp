/**
 * @file core/math/ccov_impl.hpp
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ColumnCovariance(X) is same as cov(trans(X)) but without the cost of
 * computing trans(X).
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

template<typename eT>
inline arma::Mat<eT> ColumnCovariance(const arma::Mat<eT>& x,
                                      const size_t normType)
{
  if (normType > 1)
  {
    Log::Fatal << "ColumnCovariance(): normType must be 0 or 1!" << std::endl;
  }

  arma::Mat<eT> out;

  if (x.n_elem > 0)
  {
    const arma::Mat<eT>& xAlias = (x.n_cols == 1) ?
        arma::Mat<eT>(const_cast<eT*>(x.memptr()), x.n_cols, x.n_rows, false,
            false) :
        arma::Mat<eT>(const_cast<eT*>(x.memptr()), x.n_rows, x.n_cols, false,
            false);

    const size_t n = xAlias.n_cols;
    const eT normVal = (normType == 0) ? ((n > 1) ? eT(n - 1) : eT(1)) : eT(n);

    const arma::Mat<eT> tmp = xAlias.each_col() - arma::mean(xAlias, 1);

    out = tmp * tmp.t();
    out /= normVal;
  }

  return out;
}

template<typename T>
inline arma::Mat<std::complex<T>> ColumnCovariance(
    const arma::Mat<std::complex<T>>& x,
    const size_t normType)
{
  if (normType > 1)
  {
    Log::Fatal << "ColumnCovariance(): normType must be 0 or 1" << std::endl;
  }

  using eT = std::complex<T>;

  arma::Mat<eT> out;

  if (x.is_vec())
  {
    if (x.n_rows == 1)
    {
      const arma::Mat<T> tmpMat = arma::var(trans(x), normType);
      out.set_size(1, 1);
      out[0] = tmpMat[0];
    }
    else
    {
      const arma::Mat<T> tmpMat = arma::var(x, normType);
      out.set_size(1, 1);
      out[0] = tmpMat[0];
    }
  }
  else
  {
    const size_t n = x.n_cols;
    const eT normVal = (normType == 0) ?
        ((n > 1) ? eT(n - 1) : eT(1)) : eT(n);

    const arma::Col<eT> acc = sum(x, 1);

    out = x * trans(arma::conj(x));
    out -= (acc * trans(arma::conj(acc))) / eT(n);
    out /= normVal;
  }

  return out;
}

} // namespace mlpack

#endif // MLPACK_CORE_MATH_CCOV_IMPL_HPP
