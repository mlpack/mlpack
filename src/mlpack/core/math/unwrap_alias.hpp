/**
 * @file core/math/unwrap_alias.hpp
 * @author Ryan Curtin
 *
 * Make an alias of a matrix if possible, or unwrap it if an expression is
 * given.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_UNWRAP_ALIAS_HPP
#define MLPACK_CORE_MATH_UNWRAP_ALIAS_HPP

namespace mlpack {

/**
 * If `in` is an expression, unwrap it into `m`.  If `in` is a matrix, then
 * create `m` as an alias of `in`.
 * `numCols`.
 */
template<typename MatType>
void UnwrapAlias(MatType& m, const MatType& in)
{
  MakeAlias(m, in, in.n_rows, in.n_cols);
}

template<typename MatType, typename InMatType>
void UnwrapAlias(MatType& m,
                 const InMatType& in)
{
  m = in;
}

} // namespace mlpack

#endif
