/**
 * @file core/math/safe_linalg.hpp
 * @author Omar Shrit
 * @author Ryan Curtin
 *
 * A shim around arma and coot functions to avoid confusion with standard 
 * library functions. This is necessary as we are using ADL to allow the
 * compiler to deduce which library the functions belong without the need
 * for namespace. This is mostly needed for MSVC compiler, gcc seems to
 * pass without an issue.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MAT_REDEF_HPP
#define MLPACK_CORE_MAT_REDEF_HPP

namespace mlpack {

template<typename eT>
inline arma::Mat<eT> SafeMax(const arma::Mat<eT>& A, const arma::Mat<eT>& B)
{
  return arma::max(A, B);
}

template<typename eT>
inline arma::Mat<eT> SafeMin(const arma::Mat<eT>& A, const arma::Mat<eT>& B)
{
  return arma::min(A, B);
}

#ifdef MLPACK_HAS_COOT

template<typename eT>
inline coot::Mat<eT> SafeMax(const coot::Mat<eT>& A, const coot::Mat<eT>& B)
{
  return coot::max(A, B);
}

template<typename eT>
inline coot::Mat<eT> SafeMin(const coot::Mat<eT>& A, const coot::Mat<eT>& B)
{
  return coot::min(A, B);
}

#endif

}

#endif
