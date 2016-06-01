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
  //! Default constructor
  DiscreteHilbertValue();

  /**
   * Construct this for the node tree. If the node is the root this method
   * computes the Hilbert value for each point in the tree's dataset.
   * @param node The node that stores this Hilbert value.
   */
  template<typename TreeType>
  DiscreteHilbertValue(const TreeType *tree);

  /**
   * Create a Hilbert value object by copying from the other node.
   * @param other The node from which the value will be copied.
   */
  template<typename TreeType>
  DiscreteHilbertValue(const TreeType &other);

  //! Free memory
  ~DiscreteHilbertValue();

  /**
   * Compare two points. It returns 1 if the first point is greater than
   * the second one, -1 if the first point is less than the second one and
   * 0 if the Hilbert values of the points are equal. In order to do it
   * this method computes the Hilbert values of the points.
   * @param pt1 The first point.
   * @param pt2 The second point.
   */
  template<typename ElemType>
  static int ComparePoints(const arma::Col<ElemType> &pt1,
                                                const arma::Col<ElemType> &pt2);

  /**
   * Compare two Hilbert values. It returns 1 if the first value is greater than
   * the second one, -1 if the first value is less than the second one and
   * 0 if the values are equal. This method does not compute the Hilbert values.
   * @param val1 The first point.
   * @param val2 The second point.
   */
  template<typename TreeType>
  static int CompareValues(TreeType *tree, DiscreteHilbertValue &val1,
                                                    DiscreteHilbertValue &val2);

  /**
   * Compare the largest Hilbert value of the node with the val value.
   * It returns 1 if the value of the node is greater than val,
   * -1 if the value of the node is less than val and
   * 0 if the values are equal. This method does not compute the Hilbert values.
   * @param tree Not used
   * @param val The Hilbert value to compare with.
   */
  template<typename TreeType>
  int CompareWith(TreeType *tree, DiscreteHilbertValue &val);

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value
   * of the point. It returns 1 if the value of the node is greater than
   * the value of the point, -1 if the value of the node is less than
   * the value of the point and 0 if the values are equal.
   * This method computes the Hilbert value of the point.
   * @param tree Not used
   * @param val The point to compare with.
   */
  template<typename TreeType,typename ElemType>
  int CompareWith(TreeType *tree, const arma::Col<ElemType> &pt);

  /**
   * Compare the largest Hilbert value of the node with the Hilbert value
   * of the point. It returns 1 if the value of the node is greater than
   * the value of the point, -1 if the value of the node is less than
   * the value of the point and 0 if the values are equal.
   * This method computes the Hilbert value of the point.
   * @param tree Not used
   * @param val The number of the point to compare with.
   */
  template<typename TreeType>
  int CompareWith(TreeType *tree, const size_t point);

  /**
   * Update the largest Hilbert value of the node and insert the point
   * in the local dataset if the node is a leaf.
   * @param node The node in which the point is being inserted.
   * @param point The number of the point being inserted.
   */
  template<typename TreeType>
  size_t InsertPoint(TreeType *node, const size_t point);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node being inserted.
   */
  template<typename TreeType>
  void InsertNode(TreeType *node);

  /**
   * Update the largest Hilbert value of the node and delete the point
   * from the local dataset.
   * @param node The node from which the point is being deleted.
   * @param localIndex The number of the point in the local dataset.
   */
  template<typename TreeType>
  void DeletePoint(TreeType *node, const size_t localIndex);

  /**
   * Update the largest Hilbert value of the node.
   * @param node The node from which another node is being deleted.
   * @param nodeIndex The number of the node being deleted.
   */
  template<typename TreeType>
  void RemoveNode(TreeType *node, const size_t nodeIndex);

  /**
   * Copy the largest Hilbert value and the local dataset
   * @param dst The node to which the information is being copied.
   * @param src The node from which the information is being copied.
   */
  template<typename TreeType>
  void Copy(TreeType *dst, TreeType *src);

  /**
   * Update the largest Hilbert value and the local dataset.
   * The children of the node (or the points that the node contains) should be
   * arranged according to their Hilbert values.
   * @param node The node in which the information should be updated.
   */
  template<typename TreeType>
  void UpdateLargestValue(TreeType *node);

  //! Copy the largest Hilbert value.
  DiscreteHilbertValue operator = (const DiscreteHilbertValue &val);

  //! Return the largest Hilbert value
  std::list<arma::Col<uint64_t>>::iterator LargestValue() const
  { return largestValue; }

  //! Modify the largest Hilbert value
  std::list<arma::Col<uint64_t>>::iterator &LargestValue()
  { return largestValue; }

  //! Modify the local dataset
  std::list<arma::Col<uint64_t>> *LocalDataset() { return localDataset; }
  //! Modify the dataset
  arma::Mat<uint64_t> *Dataset() { return dataset; }
 private:
  //! The dataset
  arma::Mat<uint64_t> *dataset;
  //! Indicates that the node owns the dataset
  bool ownsDataset;
  //! The local dataset
  std::list<arma::Col<uint64_t>> *localDataset;
  //! The largest Hilbert value
  std::list<arma::Col<uint64_t>>::iterator largestValue;

  /**
   * Calculate the Hilbert value of the point pt.
   * @param pt The point for which the Hilbert value should be calculated.
   */
  template<typename ElemType>
  static arma::Col<uint64_t> CalculateValue(const arma::Col<ElemType> &pt);

  /**
   * Compare two Hilbert values. It returns 1 if the first value is greater than
   * the second one, -1 if the first value is less than the second one and
   * 0 if the values are equal. This method does not compute the Hilbert values.
   * @param value1 The first value.
   * @param value2 The second value.
   */
  static int CompareValues(const arma::Col<uint64_t> &value1,
                           const arma::Col<uint64_t> &value2);
  /**
   * Returns true if the node has the largest Hilbert value.
   */
  bool HasValue();
};
} // namespace tree
} // namespace mlpack

// Include implementation
#include "discrete_hilbert_value_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_DISCRETE_HILBERT_VALUE_HPP
