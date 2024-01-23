/**
 * @file core/math/ccov.hpp
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ColumnCovariance(X) is the same as cov(trans(X)) but without the cost of
 * computing trans(X).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_CCOV_HPP
#define MLPACK_CORE_MATH_CCOV_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename eT>
inline
arma::Mat<eT>
ColumnCovariance(const arma::Mat<eT>& A, const size_t normType = 0);

template<typename T>
inline
arma::Mat<std::complex<T>>
ColumnCovariance(const arma::Mat<std::complex<T>>& A,
                 const size_t normType = 0);

} // namespace mlpack

// Include implementation
#include "ccov_impl.hpp"

#endif // MLPACK_CORE_MATH_CCOV_HPP
