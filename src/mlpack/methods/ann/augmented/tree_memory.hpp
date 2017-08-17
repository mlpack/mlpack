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

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
template<
  typename T,
  typename J = FFN<MeanSquaredError<>>,
  typename W = FFN<MeanSquaredError<>>
>
class TreeMemory {
public:
  TreeMemory(size_t size, size_t memDim, J joiner, W writer);

  void Initialize(arma::Mat<T>& leafValues);

  void Update(size_t pos, const arma::Col<T>& el);
  void Rebuild();

  arma::Col<T> Leaf(size_t index) const;
  arma::subview_col<T> Leaf(size_t index);
  arma::Col<T> Cell(size_t memIndex) const;
  arma::subview_col<T> Cell(size_t memIndex);

  inline size_t Root() const;
  inline size_t Left(size_t origin) const;
  inline size_t Right(size_t origin) const;
  inline size_t Parent(size_t child) const;
  inline size_t LeafIndex(size_t leafPos) const;

  size_t MemorySize() const { return memorySize; }
  size_t ActualMemorySize() const { return actualMemorySize; }

  void ResetParameters();

  J JoinObject() const { return joinFunction; }
  J& JoinObject() { return joinFunction; }

  W WriteObject() const { return writeFunction; }
  W& WriteObject() { return writeFunction; }

  arma::Mat<T> Stack(const arma::Mat<T>& left, const arma::Mat<T>& right) const;
private:
  arma::Mat<T> memory;
  J joinFunction;
  W writeFunction;
  size_t memorySize;
  size_t memoryDim;
  size_t actualMemorySize;
};
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "tree_memory_impl.hpp"
#endif