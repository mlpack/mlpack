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

/**
 * This class implements tree memory structure for HAM unit.
 */
template<
  typename T,
  typename J = FFN<MeanSquaredError<>>,
  typename W = FFN<MeanSquaredError<>>
>
class TreeMemory {
 public:
  /**
   * Initialize the memory components. 
   * 
   * @param size Number of available memory cells.
   * @param memDim Dimensionality of the single memory cell.
   * @param joiner The function that output the parent node value
   *               given child node values.
   * @param writer The function that outputs the new cell value
   *               given old cell value and update value.
   */
  TreeMemory(size_t size, size_t memDim, J joiner, W writer);

  /**
   * Initialize the memory contents.
   * 
   * @param leafValues The values of leaf nodes.
   */
  void Initialize(arma::Mat<T>& leafValues);

  /**
   * Update one memory cell by the given vector.
   * 
   * @param pos Index of the updated cell.
   * @param el Update value.
   */
  void Update(size_t pos, const arma::Col<T>& el);

  /**
   * Return the value of the leaf node.
   * 
   * @param index Index of the leaf node 
   *              (0 corresponds to the leftmost leaf of the tree)
   */
  arma::Col<T> Leaf(size_t index) const;
  /**
   * Modify the value of the leaf node.
   * 
   * @param index Index of the leaf node
   *              (0 corresponds to the leftmost leaf of the tree)
   */
  arma::subview_col<T> Leaf(size_t index);
  /**
   * Return the value of the memory cell.
   * 
   * @param index Index of the memory cell (0 corresponds to the root)
   */
  arma::Col<T> Cell(size_t memIndex) const;
  /**
   * Modify the value of the memory cell.
   * 
   * @param index Index of the memory cell (0 corresponds to the root)
   */
  arma::subview_col<T> Cell(size_t memIndex);

  /**
   * Return index of the root node.
   */
  inline size_t Root() const;
  /**
   * Return the index of the left child of the given node.
   * 
   * @param origin Parent node index.
   */
  inline size_t Left(size_t origin) const;
  /**
   * Return the index of the right child of the given node.
   * 
   * @param origin Parent node index.
   */
  inline size_t Right(size_t origin) const;
  /**
   * Return the index of the parent of the given node.
   * (For example, Parent(Left(x)) == Parent(Right(x)) == x as long as
   * all nodes are defined.)
   * 
   * @param origin Child node index.
   */
  inline size_t Parent(size_t child) const;
  /**
   * Return the memory index of the leaf.
   * (For example, Cell(LeafIndex(e)) == Leaf(e) as long as
   * all nodes are defined.)
   * 
   * @param leafPos The leaf index
   *                (0 corresponds to the leftmost leaf of the tree)
   * @return [description]
   */
  inline size_t LeafIndex(size_t leafPos) const;

  /**
   * Return the number of the leaf nodes used.
   */
  size_t MemorySize() const { return memorySize; }
  /**
   * Return the number of the leaf nodes allocated.
   * (For example, the inequality ActualMemorySize() >= MemorySize()
   * always holds.)
   */
  size_t ActualMemorySize() const { return actualMemorySize; }

  /**
   * Reset the memory information
   * (weights/parameters of JOIN and WRITE modules).
   */
  void ResetParameters();

  /**
   * Return the join function.
   */
  J JoinObject() const { return joinFunction; }
  /**
   * Modify the join function.
   */
  J& JoinObject() { return joinFunction; }

  /**
   * Return the write function.
   */
  W WriteObject() const { return writeFunction; }
  /**
   * Modify the write function.
   */
  W& WriteObject() { return writeFunction; }

  /**
   * A function that gets two matrices (with even number of columns)
   * as inputs and outputs a new matrix with the same number of columns
   * where the first matrix is on top of the new matrix and the second one
   * is on the bottom.
   * 
   * Example: Stack([1, 2], [3, 4]) == [[1, 2], [3, 4]].
   * 
   * @param left [description]
   * @param right [description]
   */
  arma::Mat<T> Stack(const arma::Mat<T>& left, const arma::Mat<T>& right) const;

 private:
  //! The storage for all node values.
  arma::Mat<T> memory;
  //! Join function.
  J joinFunction;
  //! Write function.
  W writeFunction;
  //! Number of cells used.
  size_t memorySize;
  //! Memory dimensionality.
  size_t memoryDim;
  //! Number of leafs allocated.
  size_t actualMemorySize;
};
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "tree_memory_impl.hpp"
#endif
