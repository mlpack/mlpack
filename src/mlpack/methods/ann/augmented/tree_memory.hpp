/**
 * @file tree_memory.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the TreeMemory class, which implements a memory structure
 * for HAMUnit.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AUGMENTED_TREE_MEMORY_HPP
#define MLPACK_METHODS_AUGMENTED_TREE_MEMORY_HPP

#include <vector>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
template<typename T, typename J, typename W>
class TreeMemory {
public:
  TreeMemory(size_t size, J joiner, W writer);

  void Initialize(std::vector<T>& leafValues);

  void Update(size_t pos, T el);

  T Get(size_t index);
  T GetCell(size_t memIndex);

  inline size_t Root();
  inline size_t Left(size_t origin);
  inline size_t Right(size_t origin);
  inline size_t Parent(size_t child);
  inline size_t LeafIndex(size_t leafPos);
private:
  std::vector<T> memory;
  J joinFunction;
  W writeFunction;
  size_t memorySize;
  size_t actualMemorySize;
};
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "tree_memory_impl.hpp"
#endif