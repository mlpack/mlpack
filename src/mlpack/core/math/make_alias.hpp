/**
 * @file core/math/make_alias.hpp
 * @author Ryan Curtin
 *
 * Make an alias of a matrix.  For sparse matrices, unfortunately no alias can
 * be made and a copy must be incurred.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_MAKE_ALIAS_HPP
#define MLPACK_CORE_MATH_MAKE_ALIAS_HPP

namespace mlpack {

/**
 * Reconstruct `m` as an alias around the memory `newMem`, with size `numRows` x
 * `numCols`.
 */
template<typename MatType>
void MakeAlias(MatType& m,
               typename MatType::elem_type* newMem,
               const size_t numRows,
               const size_t numCols,
               const bool strict = true,
               const typename std::enable_if_t<!IsCube<MatType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  m.~MatType();
  new (&m) MatType(newMem, numRows, numCols, false, strict);
}

/**
 * Reconstruct `c` as an alias around the memory` newMem`, with size `numRows` x
 * `numCols` x `numSlices`.
 */
template<typename CubeType>
void MakeAlias(CubeType& c,
               typename CubeType::elem_type* newMem,
               const size_t numRows,
               const size_t numCols,
               const size_t numSlices,
               const bool strict = true,
               const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  c.~CubeType();
  new (&c) CubeType(newMem, numRows, numCols, numSlices, false, strict);
}

/**
 * Make `m` an alias of `in`, using the given size.
 */
template<typename eT>
void MakeAlias(arma::Mat<eT>& m,
               const arma::Mat<eT>& in,
               const size_t numRows,
               const size_t numCols,
               const bool strict = true)
{
  MakeAlias(m, (eT*) in.memptr(), numRows, numCols, strict);
}

/**
 * Make `m` an alias of `in`, using the given size.
 */
template<typename eT>
void MakeAlias(arma::SpMat<eT>& m,
               const arma::SpMat<eT>& in,
               const size_t /* numRows */,
               const size_t /* numCols */,
               const bool /* strict */)
{
  // We can't make aliases of sparse objects, so just copy it.
  m = in;
}

/**
 * Clear an alias so that no data is overwritten.  This resets the matrix if it
 * is an alias (and does nothing otherwise).
 */
template<typename ElemType>
void ClearAlias(arma::Mat<ElemType>& mat)
{
  if (mat.mem_state >= 1)
    mat.reset();
}

/**
 * Clear an alias so that no data is overwritten.  This resets the matrix if it
 * is an alias (and does nothing otherwise).
 */
template<typename ElemType>
void ClearAlias(arma::SpMat<ElemType>& /* mat */)
{
  // We cannot make aliases of sparse matrices, so, nothing to do.
}

} // namespace mlpack

#endif
