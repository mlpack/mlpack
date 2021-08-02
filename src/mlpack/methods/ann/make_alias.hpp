/**
 * @file make_alias.hpp
 * @author Ryan Curtin
 *
 * Implementation of MakeAlias(), a utility function.
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

} // namespace ann
} // namespace mlpack

#endif
