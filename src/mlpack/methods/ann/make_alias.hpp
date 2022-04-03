/**
 * @file make_alias.hpp
 * @author Ryan Curtin
 *
 * Implementation of `MakeAlias()`, a utility function.  This is meant to be
 * used in `SetWeights()` calls in various layers, to wrap internal weight
 * objects as aliases around the given memory pointers.
 */
#ifndef MLPACK_METHODS_ANN_MAKE_ALIAS_HPP
#define MLPACK_METHODS_ANN_MAKE_ALIAS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

/**
 * Reconstruct `m` as an alias around the memory `newMem`, with size `numRows` x
 * `numCols`.
 */
template<typename MatType>
void MakeAlias(MatType& m,
               typename MatType::elem_type* newMem,
               const size_t numRows,
               const size_t numCols)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  m.~MatType();
  new (&m) MatType(newMem, numRows, numCols, false, true);
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
               const size_t numSlices)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  c.~CubeType();
  new (&c) CubeType(newMem, numRows, numCols, numSlices, false, true);
}

} // namespace ann
} // namespace mlpack

#endif
