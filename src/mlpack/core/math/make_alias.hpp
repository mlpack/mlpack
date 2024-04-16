/**
 * @file core/math/make_alias.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
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

template<typename eT>
struct IsCootType;

template<typename eT>
struct IsCootType
{
  constexpr static bool value = false;
};

#ifdef MLPACK_HAS_COOT

template<typename eT>
struct IsCootType<coot::Mat<eT>>
{
  constexpr static bool value = true;
};

//template<typename eT>
//struct IsCootType<coot::subview_col<eT>>
//{
  //constexpr static bool value = true;
//};

template<typename eT>
struct IsCootType<coot::Cube<eT>>
{
  constexpr static bool value = true;
};

#endif

/**
 * Reconstruct `m` as an alias around the memory `newMem`, with size `numRows` x
 * `numCols`.
 *
 * @param m The constructed matrix.
 * @param oldMat The original matrix we are constructing part from it.
 * @param offset The Start point of the constructed matrix.
 * @param numRows The number of rows of the construced matrix.
 * @param numCols The numbers or cols of the constructed matrix.
 * @param strict if true, be strict and use the same memory.
 */
template<typename InMatType,
         typename OutMatType>
void MakeAlias(OutMatType& m,
               const InMatType& oldMat,
               const size_t offset,
               const size_t numRows,
               const size_t numCols,
               const bool strict = true,
               const typename std::enable_if_t<!IsCootType<InMatType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  const typename InMatType::elem_type* newMem = oldMat.memptr() + offset;
  m.~Mat();
  new (&m) OutMatType(newMem, numRows, numCols, false, strict);
}

/**
 * Reconstruct `c` as an alias around the memory` newMem`, with size `numRows` x
 * `numCols` x `numSlices`.
 */
template<typename InCubeType,
         typename OutCubeType>
void MakeAlias(OutCubeType& c,
               const InCubeType& oldCube,
               const size_t offset,
               const size_t numRows,
               const size_t numCols,
               const size_t numSlices,
               const bool strict = true,
               const typename std::enable_if_t<!IsCootType<InCubeType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  const typename InCubeType::elem_type* newMem = oldCube.memptr() + offset;
  c.~Cube();
  new (&c) OutCubeType(newMem, numRows, numCols, numSlices, false, strict);
}

template<typename InMatType,
         typename OutMatType>
void MakeAlias(OutMatType& m,
               const InMatType& oldMat,
               const size_t offset,
               const size_t numRows,
               const size_t numCols,
               const bool strict = true,
               const typename std::enable_if_t<IsCootType<InMatType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  coot::dev_mem_t<typename InMatType::elem_type> newMem;
  newMem.cuda_mem_ptr = oldMat.get_dev_mem().cuda_mem_ptr + offset;
  m.~Mat();
  new (&m) OutMatType(newMem, numRows, numCols, false, strict);
}

/**
 * Reconstruct `c` as an alias around the memory` newMem`, with size `numRows` x
 * `numCols` x `numSlices`.
 */
//template<typename InColType,
         //typename OutColType>
//void MakeAlias(OutColType& c,
               //const InColType& oldCol,
               //const size_t offset,
               //const size_t numRows,
               //const size_t numCols,
               //const typename std::enable_if_t<IsCootType<InColType>::value>* = 0)
//{
  //// We use placement new to reinitialize the object, since the copy and move
  //// assignment operators in Armadillo will end up copying memory instead of
  //// making an alias.
  //typename InColType::elem_type* newMem = oldCol.m.get_dev_mem();
  //newMem = newMem + offset;
  //c.~Col();
  //new (&c) OutColType(newMem, numRows, numCols, false, true);
//}
/**
 * Reconstruct `c` as an alias around the memory` newMem`, with size `numRows` x
 * `numCols` x `numSlices`.
 */
template<typename InCubeType,
         typename OutCubeType>
void MakeAlias(OutCubeType& c,
               const InCubeType& oldCube,
               const size_t offset,
               const size_t numRows,
               const size_t numCols,
               const size_t numSlices,
               const bool strict = true,
               const typename std::enable_if_t<IsCootType<InCubeType>::value>* = 0)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  coot::dev_mem_t<typename InCubeType::elem_type> newMem;
  newMem = oldCube.get_dev_mem().cuda_mem_ptr;
  newMem = newMem + offset;
  c.~Cube();
  new (&c) OutCubeType(newMem, numRows, numCols, numSlices, false, strict);
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
