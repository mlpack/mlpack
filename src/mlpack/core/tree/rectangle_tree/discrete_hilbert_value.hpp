/**
 * @file discrete_hilbert_value.hpp
 * @author Mikhail Lozhnikov
 *
 * Defintion of the DiscreteHilbertValue class, a class that calculates
 * the ordering of points using the Hilbert curve.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

class DiscreteHilbertValue
{
 public:

  DiscreteHilbertValue();

  template<typename TreeType>
  DiscreteHilbertValue(const TreeType *tree);

  template<typename TreeType>
  DiscreteHilbertValue(const TreeType &other);

  ~DiscreteHilbertValue();

  template<typename ElemType>
  static int ComparePoints(const arma::Col<ElemType> &pt1,
                                                const arma::Col<ElemType> &pt2);

  template<typename TreeType>
  static int CompareValues(TreeType *tree, DiscreteHilbertValue &val1,
                                                    DiscreteHilbertValue &val2);

  template<typename TreeType>
  int CompareWith(TreeType *tree, DiscreteHilbertValue &val);

  template<typename TreeType,typename ElemType>
  int CompareWith(TreeType *tree, const arma::Col<ElemType> &pt);

  template<typename TreeType>
  size_t InsertPoint(TreeType *node, const size_t point);

  template<typename TreeType>
  void InsertNode(TreeType *node);

  template<typename TreeType>
  void DeletePoint(TreeType *node, const size_t localIndex);

  template<typename TreeType>
  void RemoveNode(TreeType *node, const size_t nodeIndex);

  template<typename TreeType>
  void Copy(TreeType *dst, TreeType *src);

  DiscreteHilbertValue operator = (DiscreteHilbertValue &val);

  std::list<arma::Col<uint64_t>>::iterator LargestValue() const
  { return largestValue; }

  std::list<arma::Col<uint64_t>> *LocalDataset() { return localDataset; }
  arma::Mat<uint64_t> *Dataset() { return dataset; }
 private:
  arma::Mat<uint64_t> *dataset;
  bool ownsDataset;
  std::list<arma::Col<uint64_t>> *localDataset;
  std::list<arma::Col<uint64_t>>::iterator largestValue;

  template<typename ElemType>
  static arma::Col<uint64_t> CalculateValue(const arma::Col<ElemType> &pt);
};
} // namespace tree
} // namespace mlpack

// Include implementation
#include "discrete_hilbert_value_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
