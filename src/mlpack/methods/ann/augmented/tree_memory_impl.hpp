/**
 * @file tree_memory_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of CopyTask class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AUGMENTED_TREE_MEMORY_IMPL_HPP
#define MLPACK_METHODS_AUGMENTED_TREE_MEMORY_IMPL_HPP

#include <cassert>

#include "tree_memory.hpp"

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
template<typename T, typename J, typename W>
TreeMemory<T, J, W>::TreeMemory(size_t size,
                                size_t memDim,
                                J joiner,
                                W writer)
{
  memorySize = size;
  memoryDim = memDim;
  // Rounding size to the next highest power of 2.
  assert(0 < size && size < static_cast<size_t>(1 << 31));
  actualMemorySize = 1 << static_cast<int>(ceil(log2((double) size)));
  assert(memorySize <= actualMemorySize);
  // Allocating enough memory to store all leaf values AND inner node values.
  memory = arma::zeros(memoryDim, 2 * actualMemorySize - 1);
  joinFunction = joiner;
  writeFunction = writer;
}

template<typename T, typename J, typename W>
inline arma::Col<T> TreeMemory<T, J, W>::Leaf(size_t index) const
{
  assert(0 <= index && index < memorySize);
  return memory.col(actualMemorySize - 1 + index);
}

template<typename T, typename J, typename W>
inline arma::subview_col<T> TreeMemory<T, J, W>::Leaf(size_t index)
{
  assert(0 <= index && index < memorySize);
  return memory.col(actualMemorySize - 1 + index);
}

template<typename T, typename J, typename W>
inline arma::Col<T> TreeMemory<T, J, W>::Cell(size_t index) const
{
  assert(0 <= index && index < memory.size());
  return memory.col(index);
}

template<typename T, typename J, typename W>
inline arma::subview_col<T> TreeMemory<T, J, W>::Cell(size_t index)
{
  assert(0 <= index && index < memory.size());
  return memory.col(index);
}

template<typename T, typename J, typename W>
inline size_t TreeMemory<T, J, W>::Root()
{
  return 0;
}

template<typename T, typename J, typename W>
inline size_t TreeMemory<T, J, W>::Left(size_t origin)
{
  return (origin << 1) + 1;
}

template<typename T, typename J, typename W>
inline size_t TreeMemory<T, J, W>::Right(size_t origin)
{
  return (origin << 1) + 2;
}

template<typename T, typename J, typename W>
inline size_t TreeMemory<T, J, W>::LeafIndex(size_t leafPos)
{
  assert(0 <= leafPos && leafPos < memorySize);
  return actualMemorySize - 1 + leafPos;
}

template<typename T, typename J, typename W>
inline size_t TreeMemory<T, J, W>::Parent(size_t child)
{
  if (child == 0) return actualMemorySize;
  return ((child + 1) >> 1) - 1;
}

template<typename T, typename J, typename W>
void TreeMemory<T, J, W>::Initialize(arma::Mat<T>& leafValues)
{
  assert(leafValues.n_cols <= memorySize);
  // First, write in the leaf nodes.
  memory.cols(actualMemorySize - 1,
              actualMemorySize - 1 + leafValues.n_cols - 1) = leafValues;
  if (actualMemorySize < 2) return;
  size_t lastWrittenIdx = actualMemorySize - 1 + leafValues.size() - 1;
  // After that, write into inner nodes as prescribed by joinFunction.
  for (size_t i = actualMemorySize - 2; i != Root(); --i)
  {
    size_t l = Left(i), r = Right(i);
    assert(l <= r);
    if (r > lastWrittenIdx) continue;
    Cell(i) = joinFunction(Cell(l), Cell(r));
  }
}

template<typename T, typename J, typename W>
void TreeMemory<T, J, W>::Update(size_t pos, arma::Col<T> el)
{
  assert(pos >= 0 && pos < memorySize);
  size_t start = actualMemorySize - 1 + pos;
  Cell(start) = writeFunction(Cell(start), el);
  while (start != Root())
  {
    start = Parent(start);
    size_t l = Left(start), r = Right(start);
    Cell(start) = joinFunction(Cell(l), Cell(r));
  }
}
} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif