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

TreeMemory<T>::TreeMemory(size_t size, J joiner, W writer) {
  memorySize = size;
  // Rounding size to the next highest power of 2.
  // WARNING: this is a hack that works only for 32-bit integers.
  // In practice we don't really want to 
  // allocate 1e12 memory cells anyway, though.
  assert(0 < size && size < static_cast<size_t>(1 << 31));
  actualMemorySize = size - 1;
  actualMemorySize |= actualMemorySize >> 1;
  actualMemorySize |= actualMemorySize >> 2;
  actualMemorySize |= actualMemorySize >> 4;
  actualMemorySize |= actualMemorySize >> 8;
  actualMemorySize |= actualMemorySize >> 16;
  actualMemorySize |= actualMemorySize >> 32;
  actualMemorySize++;
  assert(memorySize <= actualMemorySize);
  // Allocating enough memory to store all leaf values AND inner node values.
  memory.resize(2 * actualMemorySize - 1);
  joinFunction = joiner;
  writeFunction = writer;
}

inline T TreeMemory<T>::Get(size_t index) {
  assert(0 <= index && index < memorySize);
  return memory[actualMemorySize - 1 + index];
}

inline size_t TreeMemory<T>::Root() {
  return 0;
}

inline size_t TreeMemory<T>::Left(size_t origin) {
  return res = (origin << 1) + 1;
}

inline size_t TreeMemory<T>::Right(size_t origin) {
  return res = (origin << 1) + 2;
}

inline size_t TreeMemory<T>::Parent(size_t child) {
  if (child == 0) return actualMemorySize;
  return ((child + 1) >> 1) - 1;
}

void TreeMemory<T>::Initialize(vector<T>& leafValues) {
  assert(leafValues.size() <= memorySize);
  // First, write in the leaf nodes.
  for (size_t i = 0; i < leafValues.size(); ++i) {
    memory[actualMemorySize - 1 + i] = leafValues[i];
  }
  size_t lastWrittenIdx = actualMemorySize - 1 + leafValues.size() - 1;
  // After that, write into inner nodes as prescribed by writeFunction.
  for (size_t i = actualMemorySize - 2; ; --i) {
    size_t l = Left(i), r = Right(i);
    assert(l <= r);
    if (r > lastWrittenIdx) continue;
    memory[i] = joinFunction(memory[l], memory[r]);

    if (i == Root()) break;
  }
}

void TreeMemory<T>::Update(size_t pos, T el) {
  assert(pos >= 0 && pos < memorySize);
  size_t start = actualMemorySize - 1 + pos;
  memory[start] = writeFunction(memory[start], el);
  while (true) {
    start = Parent(start);
    size_t l = Left(start), r = Right(start);
    memory[start] = joinFunction(memory[l], memory[r]);
    if (start == Root()) break;
  }
}

#endif