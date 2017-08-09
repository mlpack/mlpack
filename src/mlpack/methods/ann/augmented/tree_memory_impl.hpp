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
template<typename T>
TreeMemory<T>::TreeMemory(size_t size,
                          size_t memDim,
                          LayerTypes joiner,
                          LayerTypes writer)
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

template<typename T>
inline arma::Col<T> TreeMemory<T>::Leaf(size_t index) const
{
  assert(0 <= index && index < memorySize);
  return memory.col(actualMemorySize - 1 + index);
}

template<typename T>
inline arma::subview_col<T> TreeMemory<T>::Leaf(size_t index)
{
  assert(0 <= index && index < memorySize);
  return memory.col(actualMemorySize - 1 + index);
}

template<typename T>
inline arma::Col<T> TreeMemory<T>::Cell(size_t index) const
{
  assert(0 <= index && index < memory.size());
  return memory.col(index);
}

template<typename T>
inline arma::subview_col<T> TreeMemory<T>::Cell(size_t index)
{
  assert(0 <= index && index < memory.size());
  return memory.col(index);
}

template<typename T>
inline size_t TreeMemory<T>::Root()
{
  return 0;
}

template<typename T>
inline size_t TreeMemory<T>::Left(size_t origin)
{
  return (origin << 1) + 1;
}

template<typename T>
inline size_t TreeMemory<T>::Right(size_t origin)
{
  return (origin << 1) + 2;
}

template<typename T>
inline size_t TreeMemory<T>::LeafIndex(size_t leafPos)
{
  assert(0 <= leafPos && leafPos < memorySize);
  return actualMemorySize - 1 + leafPos;
}

template<typename T>
inline size_t TreeMemory<T>::Parent(size_t child)
{
  if (child == 0) return actualMemorySize;
  return ((child + 1) >> 1) - 1;
}

template<typename T>
void TreeMemory<T>::Initialize(arma::Mat<T>& leafValues)
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
    joinFunction.Predict(Stack(Cell(l), Cell(r)), Cell(i));
  }
}

template<typename T>
void TreeMemory<T>::Update(size_t pos, arma::Col<T> el)
{
  assert(pos >= 0 && pos < memorySize);
  size_t start = actualMemorySize - 1 + pos;
  writeFunction(Stack(Cell(start), el), Cell(start));
  while (start != Root())
  {
    start = Parent(start);
    size_t l = Left(start), r = Right(start);
    joinFunction(Stack(Cell(l), Cell(r)), Cell(start));
  }
}

template<typename T>
arma::Mat<T> TreeMemory<T>::Stack(arma::Mat<T> left, arma::Mat<T> right)
{
  assert(left.n_cols == right.n_cols);
  arma::Mat<T> result(left.n_rows + right.n_rows, left.n_cols);
  result.rows(0, left.n_rows - 1) = left;
  result.rows(left.n_rows, left.n_rows + right.n_rows - 1) = right;
  return result;
}
} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif