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
 * Reconstruct `v` as an alias around the memory `newMem`, with size `numRows` x
 * `numCols`.
 */
template<typename InVecType, typename OutVecType>
void MakeAlias(OutVecType& v,
               const InVecType& oldVec,
               const size_t numElems,
               const size_t offset = 0,
               const bool strict = true,
               const typename std::enable_if_t<
                   IsVector<OutVecType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  typename InVecType::elem_type* newMem =
      const_cast<typename InVecType::elem_type*>(oldVec.memptr()) + offset;
  v.~OutVecType();
  new (&v) OutVecType(newMem, numElems, false, strict);
}

/**
 * Reconstruct `m` as an alias around the memory `newMem`, with size `numRows` x
 * `numCols`.
 */
template<typename InMatType, typename OutMatType>
void MakeAlias(OutMatType& m,
               const InMatType& oldMat,
               const size_t numRows,
               const size_t numCols,
               const size_t offset = 0,
               const bool strict = true,
               const typename std::enable_if_t<
                   IsMatrix<OutMatType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  typename InMatType::elem_type* newMem =
      const_cast<typename InMatType::elem_type*>(oldMat.memptr()) + offset;
  m.~OutMatType();
  new (&m) OutMatType(newMem, numRows, numCols, false, strict);
}

/**
 * Reconstruct `c` as an alias around the memory` newMem`, with size `numRows` x
 * `numCols` x `numSlices`.
 */
template<typename InCubeType, typename OutCubeType>
void MakeAlias(OutCubeType& c,
               const InCubeType& oldCube,
               const size_t numRows,
               const size_t numCols,
               const size_t numSlices,
               const size_t offset = 0,
               const bool strict = true,
               const typename std::enable_if_t<IsCube<OutCubeType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  typename InCubeType::elem_type* newMem =
      const_cast<typename InCubeType::elem_type*>(oldCube.memptr()) + offset;
  c.~OutCubeType();
  new (&c) OutCubeType(newMem, numRows, numCols, numSlices, false, strict);
}

/**
 * Make `m` an alias of `in`, using the given size.
 */
template<typename eT>
void MakeAlias(arma::SpMat<eT>& m,
               const arma::SpMat<eT>& in,
               const size_t /* numRows */,
               const size_t /* numCols */,
               const size_t /* offset */,
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
